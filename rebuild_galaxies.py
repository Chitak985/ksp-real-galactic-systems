#!/usr/bin/env python3
"""Rebuild galaxy barycenters and their volumetric appearance from saved data."""

import argparse
import math
from pathlib import Path
import re
import sys

from rebuild_configs import (
    CatalogError, ICRS_TO_GALACTIC, PARSEC_M, body_name, config_text,
    format_number, frozen_orbit, heliocentric_position, load_catalog,
    node_contents, number, render_orbit, render_template, repository_path,
    strip_comments, write_configs,
)

ROOT = Path(__file__).resolve().parent
G = 6.67430e-11
STANDARD_GRAVITY = 9.80665
SOLAR_MASS_KG = 1.98847e30
C_KM_S = 299792.458
YEAR_SECONDS = 365.25 * 86400
FAMILIES = (
    "elliptical", "compact_elliptical", "dwarf_elliptical", "dwarf_spheroidal",
    "lenticular", "barred_lenticular", "spiral", "weakly_barred_spiral", "barred_spiral",
)
SPHEROIDS = set(FAMILIES[:4])
FAMILY_LABELS = {family: family.replace("_", " ") for family in FAMILIES}
# Per-galaxy visual refinements may change appearance, never position, scale,
# the barycenter, or its orbit. Bounds follow the GalacticDisk loader.
RENDER_LIMITS = {
    "diskThickness": (float,0,100), "armCount": (int,0,12),
    "armWidth": (float,0.01,2), "armConcentration": (float,0,1),
    "armPitchAngle": (float,5,35), "armStartRadius": (float,0.05,0.5),
    "armIrregularity": (float,0,0.5), "interarmFraction": (float,0,1),
    "bulgeFraction": (float,0,1), "dustOpacity": (float,0,80),
    "bulgeScaleX": (float,0,100), "bulgeScaleY": (float,0,100), "bulgeScaleZ": (float,0,100),
    "barLength": (float,0,200), "barWidth": (float,0,200), "barHeight": (float,0,200),
    "dustCloudCount": (int,0,2000), "minDustCloudSize": (float,0.05,100),
    "maxDustCloudSize": (float,0.05,100), "dustLaneStrength": (float,0,4),
    "dustLaneWidth": (float,0.1,2), "dustCloudClumpiness": (float,0,1),
    "enableDustClouds": (bool,0,1), "enableStarFormation": (bool,0,1),
    "starFormingRegionCount": (int,0,1000),
    "minStarFormingRegionSize": (float,0.05,100), "maxStarFormingRegionSize": (float,0.05,100),
    "starFormationInnerRadius": (float,0,1), "starFormationOuterRadius": (float,0,1),
    "starFormingRegionIntensity": (float,0,30), "youngStarGlowIntensity": (float,0,15),
    "youngStarFraction": (float,0,1), "starBrightness": (float,0,8),
    "volumeIntensity": (float,0,50), "volumeExposure": (float,0.1,4),
}
EXCLUDED = {
    "milkyway", "namemilkyway", "sagittariusa", "andromeda", "m31", "ngc224", "andromedaa",
    "triangulum", "m33", "ngc598", "triangulumbarycenter", "ngc5866",
    "gnz11", "jadesgsz140",
}


def default_catalog():
    return {
        "schema_version": 1,
        "templates": {
            "base": "templates/galaxies/base.cfg.template",
            "common": "templates/galaxies/common.disk.template",
            "types": {family: f"templates/galaxies/{family}.disk.template" for family in FAMILIES},
        },
        "settings": {
            "epoch_seconds": 0, "frozen_period_seconds": 3e300,
            "scaled_space_factor": 6000, "soi_radius_fraction": 0.02,
            "min_soi_pc": 5, "max_soi_pc": 100,
            "cosmology": {"h0_km_s_mpc": 67.4, "omega_m": 0.315, "omega_r": 9.2e-5},
        },
        "galaxies": [], "rendering_overrides": {},
    }


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def classify(morphology, dwarf_class="", t_type=None):
    """Regular Hubble families; never guess a regular model for a peculiar object."""
    original = morphology.strip().replace(" ", "")
    morph = original.casefold()
    dwarf = dwarf_class.strip().casefold()
    if any(tag in morph for tag in ("pec", "irr", "merg", "interact", "ring")) or "(r" in morph or "(rs" in morph:
        raise CatalogError(f"Unsupported irregular, ring, merging, or peculiar morphology: {morphology}")
    if dwarf in {"ir", "im", "sm", "bcd", "hicld", "tr"} or morph.startswith(("i0", "im", "ibm", "sbm", "sm", "bcd")):
        raise CatalogError(f"Unsupported irregular/Magellanic morphology: {morphology or dwarf_class}")
    if morph.startswith(("ce", "ucd")):
        return "compact_elliptical"
    if morph.startswith(("dsph", "sph")) or dwarf == "sph":
        return "dwarf_spheroidal"
    if morph.startswith("de") or dwarf in {"de", "deem"}:
        return "dwarf_elliptical"
    if morph.startswith(("e", "ce", "cd", "dc")):
        return "elliptical"
    t = None if t_type is None else number(t_type, "T type")
    if t is None:
        try:
            t = float(morph)
        except ValueError:
            pass
    lens = morph.startswith(("s0", "sa0", "sab0", "sb0", "ds0")) or (t is not None and -3 <= t <= 0)
    if lens:
        return "barred_lenticular" if original.startswith(("SB0", "SAB0")) else "lenticular"
    # Capital B denotes a bar. Ordinary Sb/Sab stages must not become SB/SAB.
    if original.startswith("SAB"):
        return "weakly_barred_spiral"
    if original.startswith("SB"):
        return "barred_spiral"
    if morph.startswith(("sa", "sb", "sc", "sd", "s(")) or (t is not None and 0 < t < 9):
        return "spiral"
    if t is not None and t < -3:
        return "elliptical"
    raise CatalogError(f"Unrecognized regular morphology: {morphology!r}; select a reviewed template explicitly.")


def spiral_stage(galaxy):
    if galaxy.get("hubble_t") is not None:
        return max(1, min(8, number(galaxy["hubble_t"], "Hubble stage")))
    morphology = galaxy["morphology"].replace(" ", "")
    match = re.search(r"(ab|bc|cd|dm|a|b|c|d)$", morphology)
    return {"a":1,"ab":2,"b":3,"bc":4,"c":5,"cd":6,"d":7,"dm":8}.get(match.group(1) if match else "",5)


def _simpson(fn, high, steps=2048):
    step = high / steps
    return step / 3 * (fn(0) + fn(high) + math.fsum(
        (4 if i % 2 else 2) * fn(i * step) for i in range(1, steps)))


def cosmology(redshift, settings):
    """Flat FLRW distances, with radiation, for an explicitly cosmological redshift."""
    z = number(redshift, "cosmological redshift", positive=True)
    if z > 30:
        raise CatalogError("Cosmology supports redshifts up to 30.")
    h0 = number(settings["h0_km_s_mpc"], "H0", positive=True)
    matter = number(settings["omega_m"], "omega_m", positive=True)
    radiation = number(settings["omega_r"], "omega_r")
    if radiation < 0 or matter + radiation >= 1:
        raise CatalogError("Flat cosmology requires 0 < omega_m + omega_r < 1.")
    def expansion(x):
        return math.sqrt(matter * (1+x)**3 + radiation * (1+x)**4 + 1-matter-radiation)
    comoving = C_KM_S / h0 * _simpson(lambda x: 1 / expansion(x), z)
    hubble_seconds = 1e3 * PARSEC_M / h0
    lookback = hubble_seconds * _simpson(lambda x: 1 / ((1+x) * expansion(x)), z)
    return {"placement_mpc": comoving, "angular_diameter_mpc": comoving / (1+z),
            "luminosity_mpc": comoving * (1+z), "lookback_years": lookback / YEAR_SECONDS}


def distance_measures(galaxy, settings):
    distance = galaxy["distance"]
    if distance["kind"] == "cosmological_redshift":
        return cosmology(distance["cosmological_redshift"], settings["cosmology"])
    if distance["kind"] not in {"redshift_independent", "adopted_local"}:
        raise CatalogError("distance.kind must be redshift_independent, adopted_local, or cosmological_redshift.")
    d = number(distance["mpc"], "distance.mpc", positive=True)
    # A local radial velocity can be negative; it is never a cosmological clock.
    # An adopted local compilation may include group or flow-model distances;
    # do not mislabel those estimates as redshift-independent measurements.
    return {"placement_mpc": d, "angular_diameter_mpc": d, "luminosity_mpc": d,
            "lookback_years": d * 1e6 * PARSEC_M / (C_KM_S * 1000) / YEAR_SECONDS}


def sky_basis(ra_deg, dec_deg):
    ra, dec = map(math.radians, (ra_deg, dec_deg))
    return ((math.cos(dec)*math.cos(ra), math.cos(dec)*math.sin(ra), math.sin(dec)),
            (-math.sin(ra), math.cos(ra), 0),
            (-math.sin(dec)*math.cos(ra), -math.sin(dec)*math.sin(ra), math.cos(dec)))


def icrs_to_unity(vector):
    g = [math.fsum(c*v for c, v in zip(row, vector)) for row in ICRS_TO_GALACTIC]
    return (g[0], g[2], g[1])


def orientation(ra_deg, dec_deg, position_angle_deg, inclination_deg):
    """Unity Euler Z-X-Y angles; projected major axis follows PA east of north."""
    radial, east, north = sky_basis(ra_deg, dec_deg)
    pa, inc = map(math.radians, (position_angle_deg, inclination_deg))
    major = tuple(math.cos(pa)*n + math.sin(pa)*e for n, e in zip(north, east))
    minor = tuple(-math.sin(pa)*n + math.cos(pa)*e for n, e in zip(north, east))
    normal = tuple(math.cos(inc)*r + math.sin(inc)*m for r, m in zip(radial, minor))
    disk_minor = tuple(math.cos(inc)*m - math.sin(inc)*r for r, m in zip(radial, minor))
    # Swapping orbital Y/Z reflects handedness; reverse the local Z axis to
    # obtain a proper Unity rotation while retaining the same apparent ellipse.
    columns = (icrs_to_unity(major), icrs_to_unity(normal),
               tuple(-v for v in icrs_to_unity(disk_minor)))
    r = [[columns[col][row] for col in range(3)] for row in range(3)]
    x = math.asin(max(-1, min(1, -r[1][2])))
    if abs(math.cos(x)) > 1e-10:
        y, z = math.atan2(r[0][2], r[2][2]), math.atan2(r[1][0], r[1][1])
    else:
        y, z = math.atan2(-r[2][0], r[0][0]), 0
    return tuple(math.degrees(v) for v in (x, y, z))


def temperature_rgb(temperature, alpha):
    """Approximate blackbody-equivalent RGB, not a reconstructed galaxy spectrum."""
    t = max(1000, min(40000, temperature)) / 100
    red = 255 if t <= 66 else 329.698727446 * (t-60)**-0.1332047592
    green = 99.4708025861 * math.log(t)-161.1195681661 if t <= 66 else 288.1221695283*(t-60)**-0.0755148492
    blue = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231*math.log(t-10)-305.0447927307)
    return ", ".join(format_number(v) for v in [*(max(0, min(1, c/255)) for c in (red,green,blue)), alpha])


def replacements(galaxy, settings, inclination_deg=None, position_angle_deg=None):
    family = galaxy["template"]
    if family not in FAMILIES:
        raise CatalogError(f"Unsupported galaxy template: {family}")
    name = config_text(galaxy["body_name"], "body_name")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or body_name(name) != name:
        raise CatalogError("body_name must be a safe KSP identifier.")
    if canonical(name) in EXCLUDED or canonical(galaxy["name"]) in EXCLUDED:
        raise CatalogError("This galaxy is excluded or already supplied by the mod.")
    ra, dec = number(galaxy["ra_deg"], "RA"), number(galaxy["dec_deg"], "Dec")
    if not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise CatalogError("Invalid J2000 sky coordinates.")
    z = number(galaxy["redshift"], "observed redshift")
    if not -0.1 < z <= 30:
        raise CatalogError("Observed redshift must be in (-0.1, 30].")
    major = number(galaxy["angular_major_arcmin"], "angular diameter", positive=True)
    q = number(galaxy["axis_ratio"], "axis ratio", positive=True)
    if major >= 600 or q > 1:
        raise CatalogError("Angular diameter must be below 10 degrees; 0 < axis ratio <= 1.")
    pa = number(galaxy["position_angle_deg"], "position angle") % 180
    if position_angle_deg is not None:
        pa = number(position_angle_deg, "rendering position angle") % 180
    distances = distance_measures(galaxy, settings)
    tangent = math.tan(math.radians(major / 120))
    render_radius_pc = distances["placement_mpc"] * 1e6 * tangent
    physical_radius_pc = distances["angular_diameter_mpc"] * 1e6 * tangent
    # For disks, infer inclination using an explicitly assumed intrinsic thickness.
    # Spheroids instead retain the observed ellipticity, with unknown depth assumed.
    intrinsic_q = 0.25 if "lenticular" in family else 0.15
    inclination = 0 if family in SPHEROIDS else math.degrees(math.acos(math.sqrt(
        max(0, (q*q-intrinsic_q*intrinsic_q) / (1-intrinsic_q*intrinsic_q)))))
    if inclination_deg is not None:
        inclination=number(inclination_deg,"rendering inclination")
        if not 0 <= inclination <= 90:
            raise CatalogError("Rendering inclination must be between 0 and 90 degrees.")
    extent_sigma = number(galaxy.get("spheroid_extent_sigma",3.5),"spheroid extent sigma",positive=True)
    if not 2 <= extent_sigma <= 6 or ("spheroid_extent_sigma" in galaxy and family not in SPHEROIDS):
        raise CatalogError("A spheroid extent must be between 2 and 6 sigma and only applies to spheroids.")
    angles = orientation(ra, dec, pa, inclination)
    xyz = heliocentric_position({"frame":"ICRS", "epoch":"J2000", "ra_deg":ra,
                                "dec_deg":dec, "distance_pc":distances["placement_mpc"]*1e6})
    if number(settings["frozen_period_seconds"], "frozen period") != 3e300:
        raise CatalogError("Galaxies require the frozen period 3E300.")
    orbit = frozen_orbit(xyz, settings)
    mass_kg = number(galaxy["mass_solar"], "mass_solar", positive=True) * SOLAR_MASS_KG
    low, high = (number(settings[key], key, positive=True) for key in ("min_soi_pc","max_soi_pc"))
    fraction = number(settings["soi_radius_fraction"], "SOI fraction", positive=True)
    if low > high or fraction >= 0.1:
        raise CatalogError("SOI bounds invalid or SOI fraction is not small (< 0.1).")
    soi_pc = min(high, max(low, physical_radius_pc * fraction))
    if soi_pc >= physical_radius_pc * 0.1:
        raise CatalogError("SOI would encompass too much of this galaxy; reduce its bounds.")
    notes = config_text(galaxy["uncertainty_note"], "uncertainty_note")
    display = config_text(galaxy["name"], "name")
    lookback_myr = distances["lookback_years"] / 1e6
    config_text(galaxy["morphology"], "morphology")
    description = (f"{display}: {FAMILY_LABELS[family]}. "
                   f"Seen approximately {lookback_myr:.2f} million years ago (z={z:.6g}). {notes}")
    if family in SPHEROIDS:
        description += " Apparent elongation follows sky measurements; three-dimensional shape and rotation axis are uncertain."
    else:
        description += " A single plane approximates the disk; near-side choice, spiral phase, and any warp remain uncertain."
    # Catalog B, or measured V when B is unavailable, already describes received
    # light. Nearby V may explicitly be extinction-corrected. Do not apply
    # cosmological surface-brightness dimming to this photometry a second time.
    magnitude,band = galaxy.get("b_magnitude"),"B"
    if magnitude is None:
        magnitude,band = galaxy.get("v_magnitude"),"V"
    if magnitude is not None:
        area = math.pi * (major*30)**2 * q
        mu = number(magnitude,band+" magnitude") + 2.5 * math.log10(area)
        brightness = max(0.03, min(1, 0.5 * 10**(-0.4*(mu-23))))
    else:
        brightness = min(1, (1+max(0,z))**-4)
    # Observed surface brightness is a 2D quantity. It cannot be used directly
    # as the shader's 3D emissivity: a Gaussian bulge integrates much more light
    # than a thin disk. Separate profile exposures keep the disk visible and
    # avoid bleaching old populations through the shader's tone mapper.
    # These are photographic display estimates, not an absolute radiance fit.
    display_signal=math.sqrt(brightness)
    if family in SPHEROIDS:
        volume_intensity, star_brightness = 0.55*display_signal, 0.3*display_signal
    elif "lenticular" in family:
        volume_intensity, star_brightness = 2.5*display_signal, 0.2*display_signal
    else:
        volume_intensity, star_brightness = 20*display_signal, display_signal
    old_population = family in SPHEROIDS or "lenticular" in family
    stage = spiral_stage(galaxy)
    values = {
        "galaxy_name":name, "display_name":display, "description":description,
        "mass_kg":mass_kg, "soi":soi_pc*PARSEC_M,
        "gravity_g":G*mass_kg/STANDARD_GRAVITY, "orbit":render_orbit(orbit),
        "object_scale":render_radius_pc*PARSEC_M/100/number(settings["scaled_space_factor"], "scaled space factor", positive=True),
        "rotation_x":angles[0], "rotation_y":angles[1], "rotation_z":angles[2],
        "spheroid_sigma":100/extent_sigma, "spheroid_minor_sigma":100*q/extent_sigma,
        "seed":int.from_bytes(name.encode("ascii"),"little") % 2147483647,
        "brightness":brightness,
        "star_brightness":star_brightness,
        "volume_intensity":volume_intensity,
        "spiral_pitch_deg":10+2.5*(stage-1),
        "spiral_bulge_fraction":round(max(0.05, 0.35-0.045*(stage-1)),3),
        "spiral_arm_count":4 if stage>=6 else 2,
        "spiral_interarm_fraction":min(0.8, 0.35+0.06*stage),
        "bulge_color":temperature_rgb((4300 if old_population else 4400)/(1+z),0.85),
        "arms_color":temperature_rgb(9000/(1+z),0.65),
        "interarm_color":temperature_rgb((4600 if old_population else 5800)/(1+z),0.55),
        "highlight_color":temperature_rgb(11000/(1+z),0.65),
        # Approximate optical H-alpha visibility; it leaves the visible band at
        # high redshift, rather than remaining an unshifted pink optical feature.
        "nebula_color":"1, 0.035, 0.14, "+format_number(
            0.7*max(0,min(1,(780-656.28*(1+z))/80)) if 656.28*(1+z)>=380 else 0),
        "lookback_myr":lookback_myr, "redshift":z,
    }
    return {k: v if isinstance(v,str) else format_number(v) for k,v in values.items()}


def _template(root, path):
    path = repository_path(root, path)
    if path.suffix.lower() == ".cfg":
        raise CatalogError("Galaxy templates must not end in .cfg.")
    return path.read_text(encoding="utf-8-sig")


def _fragment(template, values):
    # Same strict placeholder handling as the star builder, without its Body check.
    from rebuild_configs import PLACEHOLDER
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise CatalogError(f"Unknown galaxy template placeholder: {key}")
        start = template.rfind("\n", 0, match.start()) + 1
        indent = template[start:match.start()]
        value = values[key]
        if "\n" in value and indent.strip():
            raise CatalogError("Multiline placeholders must occupy their own line.")
        return value.replace("\n", "\n"+indent) if "\n" in value else value
    text = PLACEHOLDER.sub(replace,template)
    if "{{" in text or "}}" in text:
        raise CatalogError("Unresolved galaxy template placeholder.")
    return text


def _existing_body_names(text):
    """Read Body-level names regardless of field order or nested Template names."""
    code = re.sub(r"([{}])", r"\n\1\n", strip_comments(text))
    stack, pending = [], None
    for line in code.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "{":
            stack.append(pending); pending = None
        elif line == "}":
            if stack: stack.pop()
        elif "=" in line:
            key, value = map(str.strip,line.split("=",1))
            if stack and stack[-1] == "Body" and key == "name":
                yield value
        else:
            pending = line


def apply_rendering_overrides(parameters, overrides):
    if not isinstance(overrides,dict):
        raise CatalogError("Rendering parameters must be an object.")
    for key,value in overrides.items():
        if key not in RENDER_LIMITS:
            raise CatalogError(f"Unsupported or protected rendering parameter: {key}")
        kind,low,high=RENDER_LIMITS[key]
        if kind is bool:
            if not isinstance(value,bool):
                raise CatalogError(f"{key} must be a boolean.")
            formatted=str(value).lower()
        else:
            value=number(value,key)
            if not low <= value <= high or (kind is int and not value.is_integer()):
                raise CatalogError(f"{key} is outside its supported range/type.")
            formatted=format_number(value)
        pattern=r"(?m)^(\s*"+re.escape(key)+r"\s*=)[^\n]*"
        if re.search(pattern,parameters):
            parameters=re.sub(pattern,lambda match:match.group(1)+" "+formatted,parameters)
        else:
            parameters+="\n"+key+" = "+formatted
    return parameters


def build_configs(catalog, root=ROOT, selected=None):
    root = Path(root).resolve()
    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported galaxy catalog schema.")
    if not isinstance(catalog["galaxies"], list):
        raise CatalogError("galaxies must be an array.")
    registry, settings = catalog["templates"], catalog["settings"]
    rendering=catalog.get("rendering_overrides",{})
    if not isinstance(rendering,dict):
        raise CatalogError("rendering_overrides must be an object keyed by galaxy ID.")
    selected = set(selected or ())
    identifiers, names, outputs = set(), set(), {}
    for galaxy in catalog["galaxies"]:
        identifier = config_text(galaxy["id"], "galaxy id")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise CatalogError("Galaxy IDs must be lowercase identifiers.")
        body_name(identifier)  # Also reject reserved Windows output filenames.
        name = galaxy["body_name"]
        if identifier in identifiers or name.casefold() in names:
            raise CatalogError("Duplicate galaxy identifier or body name.")
        identifiers.add(identifier); names.add(name.casefold())
        refinement=rendering.get(identifier,{})
        if not isinstance(refinement,dict) or set(refinement)-{"parameters","inclination_deg","position_angle_deg","notes","sources"}:
            raise CatalogError(f"Invalid rendering refinement for {identifier}.")
        values = replacements(galaxy,settings,refinement.get("inclination_deg"),refinement.get("position_angle_deg"))
        common = _fragment(_template(root,registry["common"]), values).strip()
        values["common_parameters"] = common
        disk = _fragment(_template(root,registry["types"][galaxy["template"]]), values).strip()
        disk = apply_rendering_overrides(disk,refinement.get("parameters",{}))
        values["disk_parameters"] = disk
        text = render_template(_template(root,registry["base"]),values)
        props, orbit, disk_node = (node_contents(text,key) for key in ("Properties","Orbit","GalacticDiskDefinition"))
        for key,expected in (("referenceBody","Sun"),("period","3E300"),("mode","OFF")):
            if not re.search(r"(?m)^\s*"+key+r"\s*=\s*"+re.escape(expected)+r"\s*$",orbit):
                raise CatalogError(f"Galaxy placement requires {key} = {expected}.")
        for key,expected in (("renderMode","Volumetric"),("rotationPeriod","0"),("alignRotationToOrbit","false")):
            if not re.search(r"(?mi)^\s*"+key+r"\s*=\s*"+expected+r"\s*$",disk_node):
                raise CatalogError(f"Galaxy renderer requires {key} = {expected}.")
        if re.search(r"(?m)^\s*\.\.\.\s*$",text) or len(re.findall(r"(?m)^\s*GalacticDiskDefinition\s*$",strip_comments(text)))!=1:
            raise CatalogError("A galaxy must generate exactly one complete disk definition.")
        for section in (props,orbit,disk_node):
            keys = re.findall(r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=",section)
            if len(keys)!=len(set(keys)):
                raise CatalogError("Duplicate keys in a generated galaxy node.")
        if not re.search(r"(?m)^\s*planet\s*=\s*"+re.escape(name)+r"\s*$",disk_node):
            raise CatalogError("Disk planet must match the galaxy body.")
        body = node_contents(text,"Body")
        if not re.search(r"(?m)^\s*name\s*=\s*"+re.escape(name)+r"\s*$",body):
            raise CatalogError("Body name must match the galaxy catalog.")
        path = repository_path(root,f"RealGalacticSystems/Systems/{identifier}.cfg")
        header = f"// Generated by rebuild_galaxies.py; edit data/galaxies.json or templates/galaxies.\n// Galaxy ID: {identifier}\n"
        if path.exists() and not path.read_text(encoding="utf-8-sig").startswith(header):
            raise CatalogError(f"Refusing to overwrite an unmanaged config: {path.name}")
        outputs[path] = header + text.rstrip() + "\n"
    if selected-identifiers:
        raise CatalogError("Unknown galaxy IDs: "+", ".join(sorted(selected-identifiers)))
    if set(rendering)-identifiers:
        raise CatalogError("Rendering refinements refer to unknown galaxy IDs: "+", ".join(sorted(set(rendering)-identifiers)))
    # Detect the same body under an existing directory or an unrelated filename.
    for path in (root/"RealGalacticSystems/Systems").rglob("*.cfg"):
        if path in outputs:
            continue
        for name in _existing_body_names(path.read_text(encoding="utf-8-sig")):
            if name.casefold() in names:
                raise CatalogError(f"Body {name} already exists in {path.relative_to(root)}")
    return {p:text for p,text in outputs.items() if not selected or p.stem in selected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=ROOT/"data/galaxies.json")
    parser.add_argument("--galaxy",action="append",help="Rebuild this saved ID (repeatable).")
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    try:
        configs=build_configs(load_catalog(args.catalog),selected=args.galaxy)
        if args.check:
            stale=[p for p,text in configs.items() if not p.exists() or p.read_text(encoding="utf-8-sig")!=text]
            for p in stale: print(f"Needs rebuilding: {p.relative_to(ROOT)}")
            if stale:return 1
            print(f"All {len(configs)} galaxy config(s) match the catalog and templates.")
        elif args.dry_run:
            for p in configs:print(f"Would rebuild: {p.relative_to(ROOT)}")
        else:
            write_configs(configs)
            for p in configs:print(f"Rebuilt: {p.relative_to(ROOT)}")
        return 0
    except (CatalogError,OSError,KeyError,TypeError,ValueError) as exc:
        print(f"Cannot rebuild galaxies: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
