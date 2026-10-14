#!/usr/bin/env python3
"""Rebuild catalogued single Sun-like stars from offline JSON and templates."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parent
PARSEC_M = 3.085677581491367e16
LIGHT_YEAR_M = 9.4607304725808e15
PLACEHOLDER = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")
# IAU Galactic axes in ICRS; x points toward the Galactic center, z north.
ICRS_TO_GALACTIC = (
    (-0.0548755604162154, -0.8734370902348850, -0.4838350155487132),
    (0.4941094278755837, -0.4448296299600112, 0.7469822444972189),
    (-0.8676661490190047, -0.1980763734312015, 0.4559837761750669),
)


class CatalogError(ValueError):
    """A catalog or template cannot be safely compiled."""


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{label} must be a number.")
    try:
        value = float(value)
    except OverflowError as exc:
        raise CatalogError(f"{label} is too large.") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise CatalogError(f"{label} must be finite" + (" and positive." if positive else "."))
    return value


def format_number(value):
    value = number(value, "Generated value")
    if value.is_integer() and abs(value) < 1e16:
        return str(int(value))
    return repr(value).replace("e+", "E").replace("e", "E")


def config_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{label} must be a nonempty string.")
    if any(char in value for char in "\r\n{}\x00") or "//" in value:
        raise CatalogError(f"{label} contains configuration syntax or a line break.")
    return value


def body_name(display_name):
    name = re.sub(r"[^A-Za-z0-9]", "_", display_name).strip("_")
    if not name or name.casefold() in {"sun", "sagittariusa"}:
        raise CatalogError("The star name must identify a new body, not Sun or SagittariusA.")
    if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", name, re.IGNORECASE):
        raise CatalogError(f"{name} is a reserved Windows filename.")
    return name


def load_catalog(path):
    def reject_constant(value):
        raise CatalogError(f"JSON contains a nonfinite number: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CatalogError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def strip_comments(text):
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def node_contents(text, name):
    text = strip_comments(text)
    matches = list(re.finditer(r"(?m)^\s*" + re.escape(name) + r"\s*\{", text))
    if len(matches) != 1:
        raise CatalogError(f"Expected exactly one {name} node.")
    start = text.index("{", matches[0].start())
    depth = 1
    for index in range(start + 1, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start + 1:index]
    raise CatalogError(f"Unclosed {name} node.")


def orbit_position(orbit, time_seconds=0):
    """Position in the right-handed orbital frame (xy plane, z normal)."""
    a = number(orbit["semiMajorAxis"], "semiMajorAxis", positive=True)
    e = number(orbit["eccentricity"], "eccentricity")
    if not 0 <= e < 1:
        raise CatalogError("Only bounded elliptical/circular placement orbits are supported.")
    mean = number(orbit.get("meanAnomalyAtEpoch", 0), "meanAnomalyAtEpoch")
    elapsed = time_seconds - number(orbit.get("epoch", 0), "epoch")
    if elapsed:
        mean += math.tau * (elapsed / number(orbit["period"], "period", positive=True))
    mean %= math.tau
    eccentric = mean
    if e and mean:
        # Bisection is robust even for eccentricities close to one.
        low, high = 0.0, math.tau
        for _ in range(64):
            eccentric = (low + high) / 2
            if eccentric - e * math.sin(eccentric) < mean:
                low = eccentric
            else:
                high = eccentric
        eccentric = (low + high) / 2
    px = a * (math.cos(eccentric) - e)
    py = a * math.sqrt(1 - e * e) * math.sin(eccentric)
    inc = math.radians(number(orbit["inclination"], "inclination"))
    lan = math.radians(number(orbit["longitudeOfAscendingNode"], "longitudeOfAscendingNode"))
    arg = math.radians(number(orbit["argumentOfPeriapsis"], "argumentOfPeriapsis"))
    x = math.cos(arg) * px - math.sin(arg) * py
    y = math.sin(arg) * px + math.cos(arg) * py
    return (
        math.cos(lan) * x - math.sin(lan) * math.cos(inc) * y,
        math.sin(lan) * x + math.cos(lan) * math.cos(inc) * y,
        math.sin(inc) * y,
    )


def heliocentric_position(position):
    if position.get("frame") != "ICRS" or position.get("epoch") != "J2000":
        raise CatalogError("Positions must use ICRS at J2000; no epoch propagation is performed.")
    ra = number(position["ra_deg"], "ra_deg")
    dec = number(position["dec_deg"], "dec_deg")
    distance = number(position["distance_pc"], "distance_pc", positive=True) * PARSEC_M
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise CatalogError("RA must be in [0, 360), declination in [-90, 90].")
    ra, dec = math.radians(ra), math.radians(dec)
    icrs = (distance * math.cos(dec) * math.cos(ra),
            distance * math.cos(dec) * math.sin(ra), distance * math.sin(dec))
    return tuple(math.fsum(c * v for c, v in zip(row, icrs)) for row in ICRS_TO_GALACTIC)


def frozen_orbit(position, settings):
    radius = number(math.hypot(*position), "Placement distance", positive=True)
    x, y, z = position
    # Put the point at anomaly zero on an arbitrary polar circular orbit.
    # Its direction is encoded entirely in LAN and argument of periapsis.
    return {
        "referenceBody": "Sun",
        "semiMajorAxis": radius,
        "eccentricity": 0.0,
        "inclination": 90.0,
        "longitudeOfAscendingNode": math.degrees(math.atan2(y, x)) % 360,
        "argumentOfPeriapsis": math.degrees(math.atan2(z, math.hypot(x, y))) % 360,
        "meanAnomalyAtEpoch": 0.0,
        "epoch": number(settings["epoch_seconds"], "epoch_seconds"),
        "period": number(settings["frozen_period_seconds"], "frozen_period_seconds", positive=True),
        "mode": "OFF",
    }


def render_orbit(orbit):
    lines = ["Orbit", "{"]
    for key, value in orbit.items():
        lines.append(f"    {key} = {value if isinstance(value, str) else format_number(value)}")
    return "\n".join(lines + ["}"])


def repository_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise CatalogError("Catalog paths must be relative to the repository.")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise CatalogError(f"Path escapes the repository: {relative}")
    return path


def replacements(body, settings):
    display = config_text(body["name"], "name")
    description = config_text(body["description"], "description")
    if len(description) >= 100:
        raise CatalogError("description must contain fewer than 100 characters.")
    radius = number(body["radius_m"], "radius_m", positive=True)
    mass = number(body["mass_kg"], "mass_kg", positive=True)
    temperature = number(body["temperature_k"], "temperature_k", positive=True)
    rotation = number(body["rotation_period_days"], "rotation_period_days", positive=True) * 86400
    luminosity = number(body["luminosity_solar"], "luminosity_solar", positive=True)
    color = body["main_color"]
    if not isinstance(color, list) or len(color) != 4:
        raise CatalogError("main_color must contain four RGBA components.")
    color = [number(v, "main_color") for v in color]
    if any(not 0 <= v <= 1 for v in color):
        raise CatalogError("main_color components must be between zero and one.")
    location = heliocentric_position(body["position"])
    orbit = frozen_orbit(location, settings)
    distance_ly = number(body["position"]["distance_pc"], "distance_pc", positive=True) * PARSEC_M / LIGHT_YEAR_M
    values = {
        "body_name": body_name(display),
        "display_name": display,
        "description": description,
        "radius_m": radius,
        "mass_kg": mass,
        "rotation_period_seconds": rotation,
        "luminosity": luminosity * number(settings["sun_luminosity"], "sun_luminosity", positive=True),
        "temperature_k": temperature,
        "recovery_value": math.ceil(50 + distance_ly),
        "flying_altitude_threshold": radius / 10000,
        "space_altitude_threshold": radius * 3,
        "main_color": ",".join(format_number(v) for v in color),
        "ambient_color": ",".join(format_number(v) for v in [*(v / 10 for v in color[:3]), color[3]]),
        "orbit": render_orbit(orbit),
    }
    return {key: value if isinstance(value, str) else format_number(value) for key, value in values.items()}


def render_template(template, values):
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise CatalogError(f"Unknown template placeholder: {key}")
        value = values[key]
        if "\n" in value:
            line_start = template.rfind("\n", 0, match.start()) + 1
            indent = template[line_start:match.start()]
            if indent.strip():
                raise CatalogError("Multiline placeholders must occupy their own line.")
            return value.replace("\n", "\n" + indent)
        return value

    rendered = PLACEHOLDER.sub(replace, template)
    if "{{" in rendered or "}}" in rendered:
        raise CatalogError("Unresolved or malformed template placeholder.")
    code = strip_comments(rendered)
    depth = 0
    for char in code:
        depth += (char == "{") - (char == "}")
        if depth < 0:
            raise CatalogError("Unbalanced template braces.")
    if depth:
        raise CatalogError("Unbalanced template braces.")
    if re.search(r"\b(?:NAME_UNDERSCORES|NAME|DESC|NaN|Infinity)\b", code):
        raise CatalogError("Template contains an old placeholder or nonfinite value.")
    if len(re.findall(r"(?m)^\s*Body\s*$", code)) != 1:
        raise CatalogError("A star template must generate exactly one Body.")
    orbit = node_contents(code, "Orbit")
    if not re.search(r"(?m)^\s*referenceBody\s*=\s*Sun\s*$", orbit):
        raise CatalogError("Generated stars must orbit Sun.")
    return rendered


def build_configs(catalog, root=REPO_ROOT, selected=None):
    """Validate and render every selected body before any config is written."""
    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported catalog schema_version; expected 1.")
    settings = catalog["settings"]
    entries = catalog["bodies"]
    if not isinstance(entries, list):
        raise CatalogError("bodies must be a JSON array.")
    identifiers, names = set(), set()
    output = {}
    selected = set(selected or ())
    for body in entries:
        identifier = config_text(body["id"], "id")
        name = body_name(config_text(body["name"], "name"))
        if identifier in identifiers or name.casefold() in names:
            raise CatalogError(f"Duplicate body ID or filename: {identifier}")
        identifiers.add(identifier)
        names.add(name.casefold())
        if body["galaxy"] != "MilkyWay" or body["template"] != "sun_like" or body["system_type"] != "single":
            raise CatalogError(f"{identifier}: only single Sun-like stars in MilkyWay are supported.")
        if selected and identifier not in selected:
            continue
        template_path = repository_path(root, catalog["templates"][body["template"]])
        if template_path.suffix.casefold() == ".cfg":
            raise CatalogError("Templates must not end in .cfg.")
        values = replacements(body, settings)
        text = render_template(template_path.read_text(encoding="utf-8-sig"), values)
        body_node = node_contents(text, "Body")
        if not re.search(r"(?m)^\s*name\s*=\s*" + re.escape(name) + r"\s*$", body_node):
            raise CatalogError("Template must use {{body_name}} as its Body name.")
        path = repository_path(root, f"RealGalacticSystems/Systems/{body['galaxy']}/Systems/{name.lower()}.cfg")
        expected_parent = repository_path(root, f"RealGalacticSystems/Systems/{body['galaxy']}/Systems")
        if path.parent != expected_parent:
            raise CatalogError("Generated output must remain directly in its galaxy's Systems directory.")
        header = f"// Generated by rebuild_configs.py; edit data/bodies.json or the template.\n// Body ID: {identifier}\n"
        output[path] = header + text.rstrip() + "\n"
    if selected - identifiers:
        raise CatalogError("Unknown body IDs: " + ", ".join(sorted(selected - identifiers)))
    return output


def write_configs(configs):
    for path, text in configs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                             prefix=path.stem + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data/bodies.json")
    parser.add_argument("--body", action="append", help="Rebuild only this catalog body ID (repeatable).")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Verify saved configs match their templates and data.")
    modes.add_argument("--dry-run", action="store_true", help="Validate and list outputs without writing.")
    args = parser.parse_args(argv)
    try:
        configs = build_configs(load_catalog(args.catalog), selected=args.body)
        if args.check:
            stale = [p for p, text in configs.items() if not p.exists() or p.read_text(encoding="utf-8-sig") != text]
            for path in stale:
                print(f"Needs rebuilding: {path.relative_to(REPO_ROOT)}")
            if stale:
                return 1
            print(f"All {len(configs)} generated config(s) match the catalog and templates.")
        elif args.dry_run:
            for path in configs:
                print(f"Would rebuild: {path.relative_to(REPO_ROOT)}")
        else:
            write_configs(configs)
            for path in configs:
                print(f"Rebuilt: {path.relative_to(REPO_ROOT)}")
        return 0
    except (CatalogError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Cannot rebuild configs: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    main()
    input("Press Enter to continue...")
