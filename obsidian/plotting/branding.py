"""
Creates a set of branding color palettes and continuous color maps.

This file can be imported in other files to access branding colors through
commands such as:
    "color = obsidian_colors.teal"
    or
    "cmap = obsidian_colors.cm.obsidian_viridis"

This style is particularly convenient for usage with matplotlib.
"""

from matplotlib.colors import LinearSegmentedColormap


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/256 for i in range(0, lv, lv // 3))


class ColorScheme():
    pass


obsidian_colors = ColorScheme()


class Palette():
    pass


primary = Palette()
primary.teal = '#00857C'
primary.white = '#FFFFFF'

secondary = Palette()
secondary.blue = '#0C2340'
secondary.light_teal = '#6ECEB2'
secondary.off_white = '#F7F7F7'

accent = Palette()
accent.lime = '#BFED33'
accent.lemon = '#FFF063'
accent.pastel_blue = '#69B8F7'
accent.vista_blue = '#688CE8'
accent.rich_blue = '#5450E4'

magenta = '#d04495'

obsidian_colors.primary = primary
obsidian_colors.secondary = secondary
obsidian_colors.accent = accent

# Wrap all colors also directly into obsidian_colors unsorted by type
for attr in dir(obsidian_colors):
    if isinstance(getattr(obsidian_colors, attr), Palette):
        palette = getattr(obsidian_colors, attr)
        for sub_attr in dir(palette):
            if isinstance(getattr(palette, sub_attr), str):
                if '#' in getattr(palette, sub_attr):
                    setattr(obsidian_colors, sub_attr, getattr(palette, sub_attr))


class ColorMaps():
    pass


obsidian_cm = ColorMaps()
obsidian_cm.obsidian_viridis = LinearSegmentedColormap.from_list('obsidianViridis',
                                                                 colors=[hex_to_rgb(obsidian_colors.accent.rich_blue),
                                                                         hex_to_rgb(obsidian_colors.primary.teal),
                                                                         hex_to_rgb(obsidian_colors.accent.lemon)])

obsidian_cm.obsidian_plasma = LinearSegmentedColormap.from_list('obsidianPlasma',
                                                                colors=[hex_to_rgb(obsidian_colors.accent.rich_blue),
                                                                        hex_to_rgb(magenta),
                                                                        hex_to_rgb(obsidian_colors.accent.lemon)])

obsidian_cm.obsidian_mako = LinearSegmentedColormap.from_list('obsidianMako',
                                                              colors=[hex_to_rgb(obsidian_colors.blue),
                                                                      hex_to_rgb(obsidian_colors.teal),
                                                                      hex_to_rgb(obsidian_colors.light_teal)])

obsidian_cm.obsidian_tealshade = LinearSegmentedColormap.from_list('obsidianTealShade',
                                                                   colors=[hex_to_rgb(obsidian_colors.light_teal),
                                                                           hex_to_rgb(obsidian_colors.teal)])

obsidian_cm.obsidian_blueshade = LinearSegmentedColormap.from_list('obsidianBlueShade',
                                                                   colors=[hex_to_rgb(obsidian_colors.vista_blue),
                                                                           hex_to_rgb(obsidian_colors.rich_blue)])

obsidian_colors.cm = obsidian_cm
