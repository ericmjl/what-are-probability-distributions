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
To keep the pedagogy clean, I'm going to borrow some machinery
from the SciPy stats library.
""")

st.markdown("""
### Initialization

The Normal distribution is canonically initialized with two parameters,
`mu` and `sigma`.
`mu` controls the "central tendency" of the distribution,
i.e. where on the number line it is centered on,
while `sigma` controls the "spread" of the distribution,
i.e. how wide the Normal distribution is.
""")

with st.echo():
    from scipy.stats import norm
    class Normal:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
            self.dist = norm(loc=mu, scale=sigma)

st.markdown("""
## Probability Density Function

A probability distribution also contains
one function called the "probability density/mass function"
for continuous and discrete distributions respectively.

By definition, __area__ under the PDF/PMF must sum to 1.

This function has a characteristic shape,
such as the bell shape for a Gaussian,
and is defined over the number line for all valid values of the distribution
(also known as the support).

For example, the Gaussian has "support" from $-\infty$ to $+\infty$,
so the distribution is valid
for values of from negative to positive infinity.

The `logpdf` is nothing more than the log transform of the `pdf`.
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

st.markdown("""
## Drawing Numbers

The final piece we'll introduce is the ability to draw numbers from the distribution.

More pedantically, we're drawing numbers from the _support_ of the distribution,
i.e. only values that are valid for the distribution.
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
            """Draw n values from the distribution."""
            return self.dist.rvs(n)

st.markdown("""
Enough said!

Go ahead and configure a Gaussian using the widgets on the sidebar!
""")

st.sidebar.markdown("## Configure your gaussian")
mu = st.sidebar.slider("mu: central tendency", min_value=-5., max_value=5., value=0., step=0.1)
sigma = st.sidebar.slider("sigma: spread", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
st.sidebar.markdown("------")
gaussian = Normal(mu, sigma)

import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian(gaussian):
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
    ax.set_xlabel("support (or 'x')")
    return fig, ax, minval, maxval

fig, ax, minval, maxval = plot_gaussian(gaussian)
st.pyplot(fig)


"""### Probability and Likelihood

As mentioned above,
the Normal distribution has a probability density function.
That's the blue curve you see in the plot.
By definition,
the area under the probability density/mass function sums to one,
and so for a continuous distribution like the Gaussian,
"probability" is strictly defined _only_ for ranges,
and not for individual values.

Go ahead and calculate the total probability
for a range of x-values.
In particular, see how changing the width
changes the total probability of the x-values.
(In particular, bring the width to 0!)
"""

lower_quartile, upper_quartile = gaussian.dist.ppf([0.25, 0.75])
minrange, maxrange = st.slider("Range on the x-axis", min_value=minval, max_value=maxval, value=[lower_quartile, upper_quartile])
total_probability = gaussian.dist.cdf(maxrange) - gaussian.dist.cdf(minrange)

xs = np.linspace(minrange, maxrange, 1000)
fill_data = pd.DataFrame({
    "x": xs,
    "pdf": gaussian.pdf(xs),
    "lowerbound": np.zeros(len(xs))
})

fig2, ax2, minval, maxval = plot_gaussian(gaussian)
ax2.set_title("Probability of a range")

ax2.fill_between(fill_data["x"], fill_data["lowerbound"], fill_data["pdf"], color="red")
st.pyplot(fig2)
st.markdown(f"The total probability of that range of values is {total_probability:.2f}.")

st.markdown(f"""
On the other hand, we also use the probability density function
as a way of expressing "likelihood".
What's "likelihood"?
Basically, it's the "credibility points" idea
associated with a single number on the x-axis.
The higher the likelihood, the greater the amount of credibility points.

Likelihood, as expressed by the probability density function,
is not the **area** under the curve,
but the **height** of the curve.
As such, we can talk about the **likelihood of a single point**.
Go try it out below!
""")

xs = np.linspace(minrange, maxrange, 1000)

fig2, ax2, minval, maxval = plot_gaussian(gaussian)
x_value = st.slider("Pick a value x to caclualte its likelihood:", min_value=minval, max_value=maxval, value=(minval + maxval) / 2, step=(maxval - minval)/100)
like = gaussian.pdf(x_value)
ax2.vlines(x=x_value, ymin=0, ymax=like, color="red")
ax2.set_title(f"Likelihood: {like:.2f}")
st.pyplot(fig2)

st.markdown("""
These two demos hopefully make clear to you that we can never really speak of
the "probability of a number" in a continuous distribution.
However, we can talk about the "likelihood of a number".

### Drawing numbers from a probability distribution

Turns out, one other thing you can do with probability distributions
is to draw numbers from them.
Yes, they are random number generators!
(More generally, they are also called "generative models" of data.)
We can generate a sequence of draws
all we have to do is call on the `.draw()` method:

```python
xs = distribution.draw(1000)
```

Go ahead and request for a number of "draws" from your Gaussian!
""")

num_draws = st.number_input("Number of draws", min_value=0, max_value=2000, value=0, step=250)
draws = gaussian.draw(num_draws)

fig3, ax3, minval, maxval = plot_gaussian(gaussian)
ax3.vlines(x=draws, ymin=0, ymax=gaussian.pdf(draws), alpha=0.1)

st.pyplot(fig3)

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

Previously, you saw "probability" as the area under the curve.
"Likelihood", however, is the _height_ of the curve.
We can calculate the likelihood of a single data point
by taking its value on the x-axis and drawing a line up to the height of the curve.
The total likelihood of multiple data points
is the product of all of their likelihoods.
That said, the numbers can get small really quickly on a computer,
so to prevent underflow, we typically take the sum of log likelihoods
rather than the product of likelihoods.

Using our distribution object above, it literally is nothing more than calling:

```python
distribution.logpdf(xs).sum()
```

Let's play a game around this point.

I've got data for you, in the form of "draws" from a Gaussian.
However, I'm not telling you what the configuration of that Gaussian is.
Can you infer what the most likely configuration is?

Here's the values:
""")

data = [0.89, 0.81, 0.54, 0.50, -0.11, 1.19]
st.write(data)

st.markdown("And here's a Gaussian for you to play with. (See the sidebar.)")
mu = st.number_input("mu", min_value=-5., max_value=5., value=0., step=0.1)
sigma = st.number_input("sigma", min_value=0.1, max_value=10., value=1.0, step=0.1)
gaussian = Normal(mu, sigma)

fig4, ax4 = plt.subplots()
boundwidth = 0.001
minval, maxval = gaussian.dist.ppf([boundwidth, 1 - boundwidth])
xs = np.linspace(minval, maxval, 1000)
ys = gaussian.pdf(xs)
ax4.plot(xs, ys)
likelihoods = gaussian.pdf(data)
loglikes = gaussian.logpdf(data)
ax4.vlines(x=data, ymin=0, ymax=likelihoods)
ax4.scatter(x=data, y=likelihoods, color="red", marker="o")
ax4.set_ylim(0, max(ys) * 2)
ax4.set_title(f"Total log-likelihood: {loglikes.sum():.3f}")

st.pyplot(fig4)

st.markdown(
"""
If you tried to maximize the log likelihood,
then you were attempting "maximum likelihood estimation"!
This is a natural way of approaching estimation of parameters
given data.
""")

st.markdown(
"""
## What more from here?

As I wrote above, I have adopted Bayesian methods in my day-to-day work at NIBR.
Everything shown in this app has a natural extension to Bayesian methods.
If you're curious how Bayesian methods work computationally,
check out [this essay that I wrote][compbayes].

[compbayes]: https://ericmjl.github.io/essays-on-data-science/machine-learning/computational-bayesian-stats/
"""
)

st.sidebar.markdown("""
Made with ❤️ using matplotlib, numpy, scipy, and streamlit.

Check out the source of this app [on GitHub][gh].

Copyright 2020-present, Eric J. Ma.

[gh]: https://github.com/ericmjl/what-are-probability-distributions
""")



st.markdown(
"""
If you made it this far, congratulations!

If you liked this content and want to support my efforts creating
programming-oriented educational material for data scientists,
then please [send me a cup of coffee on Patreon][patreon]!

[patreon]: https://www.patreon.com/ericmjl

Meanwhile, I send out a programmer-oriented data science newsletter every month.
If you'd like to receive a curated newsletter of tools, tips and techniques,
come sign up at [TinyLetter][tinyletter]

[tinyletter]: https://tinyletter.com/ericmjl
"""
)

finished = st.button("Click me!")
if finished:
    st.balloons()
