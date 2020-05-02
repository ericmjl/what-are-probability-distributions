import streamlit as st
import pandas as pd

st.title("A Careful Walk Through Probability Distributions with Python")

st.markdown("""
_By Eric J. Ma, for PyCon 2020_

Hey there! Thanks for stopping by.

I made this streamlit app to explain what probability distributions are.
It's part of my talk delivered at the virtual PyCon 2020 conference.
I hope you enjoy it!

------
""")

st.header("Probability distributions as objects")

st.markdown("""
For a Pythonista, the best way to think about probability distributions
is as a Python object.
It's an object that can be configured when instantiated,
and we can do a number of things with it.

Let's think about it in concrete terms
by studying the **Normal** or **Gaussian** distribution.
""")

with st.echo():
    from scipy.stats import norm
    class Normal:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.dist = norm(loc=mu, scale=sigma)

        def pdf(self, x):
            """Evalute likelihood of data `x` given the distribution."""
            return self.dist.pdf(x)

        def logpdf(self, x):
            """Evaluate log likelihood of data `x` given the distribution."""
            return self.dist.logpdf(x)

        def draw(self, n):
            """Draw `n` values from the distribution."""
            return self.dist.rvs(n)

st.markdown("""
Let's unpack a bit of stuff here.

### Initialization

The Normal distribution is canonically initialized with two parameters,
`mu` and `sigma`.
`mu` controls the "central tendency" of the distribution,
i.e. where on the number line it is centered on,
while `sigma` controls the "spread" of the distribution,
i.e. how wide the Normal distribution is.

Go ahead and configure an initialization of a Gaussian, using the sidebar!
""")

st.sidebar.markdown("## Configure your gaussian")
mu = st.sidebar.slider("mu: central tendency", min_value=-5., max_value=5., value=0., step=0.1)
sigma = st.sidebar.slider("sigma: spread", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
st.sidebar.markdown("------")
gaussian = Normal(mu, sigma)

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


boundwidth = 0.001
minval, maxval = gaussian.dist.ppf([boundwidth, 1 - boundwidth])
xs = np.linspace(minval, maxval, 1000)
ys = gaussian.pdf(xs)
data = pd.DataFrame({"x": xs, "pdf": ys})

ax = data.plot(x="x", y="pdf", ax=ax)
ax.set_xlim(data["x"].min() - 1, data["x"].max() + 1)
ax.set_ylim(0, data["pdf"].max() * 2)
ax.set_ylabel("Probability density function")
ax.set_title("Your bespoke Gaussian!")

st.pyplot(fig)
# chart
# st.write(chart)


"""### Probability density function (and its logarithmic transform)

The Normal distribution has a probability density function.
The area under the probability density function sums to one,
and so for a continuous distribution like the Gaussian,
"probability" is strictly defined _only_ for ranges.

Go ahead and calculate the total probability
for a range of x-values.
In particular, see how changing the width
changes the total probability of the data.
"""

lower_quartile, upper_quartile = gaussian.dist.ppf([0.25, 0.75])
minrange, maxrange = st.sidebar.slider("Range on the x-axis", min_value=minval, max_value=maxval, value=[lower_quartile, upper_quartile])
total_probability = gaussian.dist.cdf(maxrange) - gaussian.dist.cdf(minrange)

xs = np.linspace(minrange, maxrange, 1000)
fill_data = pd.DataFrame({
    "x": xs,
    "pdf": gaussian.pdf(xs),
    "lowerbound": np.zeros(len(xs))
})
ax.fill_between(fill_data["x"], fill_data["lowerbound"], fill_data["pdf"], color="red")
st.pyplot(fig)
st.markdown(f"The total probability of that range of values is {total_probability:.2f}.")

st.markdown("""
### Drawing numbers from a probability distribution

Turns out, one other thing you can do with probability distributions
is to draw numbers from them.
Yes, they are random number generators!
(More generally, they are also called "generative models" of data.)

Go ahead and request for a number of "draws" from your Gaussian!
""")

st.sidebar.markdown("-----")
num_draws = st.sidebar.number_input("Number of draws", min_value=0, max_value=2000, value=0, step=20)
draws = gaussian.draw(num_draws)


ax.vlines(x=draws, ymin=0, ymax=gaussian.pdf(draws), alpha=0.1)

st.pyplot(fig)

st.markdown("""
Notice how as you increase the number of draws,
the density of draws gets in the regions that have
a higher PDF.
This corresponds to having a higher "density" of points
being drawn in that region.

-------
""")

st.markdown("""
## Inferring parameter values given data.

Let's play a game.

I've got data for you, in the form of "draws" from a Gaussian.
However, I'm not telling you what the configuration of that Gaussian is.
Can you infer what the most likely configuration is?

Here's the values:
""")

data = [0.89, 0.81, 0.54, 0.50, -0.11, 1.19]
st.write(data)

st.markdown("And here's a Gaussian for you to play with. (See the sidebar.)")
st.sidebar.markdown("## Configure the Gaussian to infer parameters")
mu = st.sidebar.number_input("mu", min_value=-5., max_value=5., value=0., step=0.1)
sigma = st.sidebar.number_input("sigma", min_value=0.1, max_value=10., value=1.0, step=0.1)
gaussian = Normal(mu, sigma)

fig, ax = plt.subplots()
boundwidth = 0.001
minval, maxval = gaussian.dist.ppf([boundwidth, 1 - boundwidth])
xs = np.linspace(minval, maxval, 1000)
ys = gaussian.pdf(xs)
ax.plot(xs, ys)
likelihoods = gaussian.pdf(data)
loglikes = gaussian.logpdf(data)
ax.vlines(x=data, ymin=0, ymax=likelihoods)
ax.set_ylim(0, max(ys) * 2)
ax.set_title(f"Total log-likelihood: {loglikes.sum():.3f}")

st.pyplot(fig)

st.markdown(
"""
If you tried to maximize the log likelihood,
then you were attempting "maximum likelihood estimation"!
It's quite a natural thing to do.
""")

st.markdown(
"""
## What more from here?

As I wrote above, I have adopted Bayesian methods in my day-to-day work at NIBR.
Everything shown in this app has a natural extension to Bayesian methods.
"""
)
