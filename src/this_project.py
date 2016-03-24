from matplotlib import cm

# colormaps
CMAP_GREY = cm.binary
CMAP_GREY.set_under("grey")

try:
    CMAP = cm.viridis
except AttributeError:
    CMAP = cm.YlGnBu
CMAP.set_under("grey")
