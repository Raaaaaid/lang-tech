import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import circlify

sent140 = pd.read_csv("./lang-tech/dataset/Sentiment140/testdata.manual.2009.06.14.csv", encoding = "ISO-8859-1", header=None,
    names=["polarity", "id", "date", "query", "user", "text"])
sent140 = sent140.replace({0:-1, 2:0, 4:1})
print(sent140.head())
group140 = sent140.groupby(['query'], sort=False)
queryGroup = group140["query"].count()
polarityGroup = group140["polarity"].sum()

### PLOT CODE ###
circles = circlify.circlify(
    queryGroup.tolist(), 
    show_enclosure=False, 
    target_enclosure=circlify.Circle(x=0, y=0, r=1)
)

labels = queryGroup.index

colormap = cm.get_cmap("bwr_r", polarityGroup.nunique())
upper = max(abs(polarityGroup.min()), polarityGroup.max())
lower = -1*max(abs(polarityGroup.min()), polarityGroup.max())
norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=lower, vmax=upper)

# Create just a figure and only one subplot
fig, ax1 = plt.subplots(figsize=(15,12))

ax1.axis("off")

# Find axis boundaries
lim = max(
    max(
        abs(circle.x) + circle.r,
        abs(circle.y) + circle.r,
    )
    for circle in circles
)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)

for circle, label, polarity in zip(circles, labels, polarityGroup.tolist()):
    #print(circle.r, label, polarity)
    x, y, r = circle
    ax1.add_patch(plt.Circle((x, y), r, alpha=0.9, facecolor=colormap(norm(polarity)), linewidth=1, edgecolor="black"))
    if (circle.r >= 0.075):
        fs = 'medium'
    else:
        fs = 'x-small'
    plt.annotate(label, (x,y), va='center', ha='center', fontsize=fs)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax1, orientation='vertical', label="Weigth of positive/negative Tweets regarding this topic", shrink=0.8)
ax1.set_title("Topic Overview of the Sentiment140 Test Data", pad=10)
fig.savefig("./lang-tech/plots/sent140TestData.png", format="png")
plt.show()