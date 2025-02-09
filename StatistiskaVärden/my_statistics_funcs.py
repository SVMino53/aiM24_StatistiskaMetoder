import math
import matplotlib.pyplot as plt
from scipy.special import hyp2f1

from typing import Callable

import scipy.special



def mean(elements: list[int | float]) -> float:
    return sum(elements)/len(elements)
def quartile(q: int, elements: list[int | float]) -> float:
    elements_sorted = sorted(elements)
    n = len(elements)
    pos = (n + 1)*q/4 - 1
    return elements_sorted[int(pos)]*(1 + int(pos) - pos) + elements_sorted[int(pos) + 1]*(pos - int(pos))
def median(elements: list[int | float]) -> float:
    return quartile(2, elements=elements)
def q1(elements: list[int | float]) -> float:
    return quartile(1, elements=elements)
def q2(elements: list[int | float]) -> float:
    return quartile(2, elements=elements)
def q3(elements: list[int | float]) -> float:
    return quartile(3, elements=elements)

def variance(elements: list[int | float]) -> float:
    m = mean(elements=elements)
    n = len(elements)
    return sum([(x - m)**2 for x in elements])/(n - 1)
def pvariance(elements: list[int | float]) -> float:
    m = mean(elements=elements)
    n = len(elements)
    return sum([(x - m)**2 for x in elements])/n
def standard_dev(elements: list[int | float]) -> float:
    return math.sqrt(variance(elements=elements))
def pstandard_dev(elements: list[int | float]) -> float:
    return math.sqrt(pvariance(elements=elements))
def mad(elements: list[int | float]) -> float:
    m = mean(elements=elements)
    n = len(elements)
    return sum([abs(x - m) for x in elements])/n

def t_pdf(df: int, t: float) -> float:
    return math.gamma((df + 1)/2)/(math.sqrt(math.pi*df)*math.gamma(df/2))*(1 + t**2/df)**(-(df + 1)/2)
def t_cdf(df: int, t: float) -> float:
    pass

class DiscreteDist:
    def __init__(self, weights: list[int | float]) -> None:
        total_weight = sum(weights)
        self.weights = [weight/total_weight for weight in weights]
    def expected(self) -> float:
        return sum([x*p for x, p in enumerate(self.weights)])
    def variance(self) -> float:
        return sum([(x - self.expected())**2*p for x, p in enumerate(self.weights)])
    def standard_dev(self) -> float:
        return math.sqrt(self.variance())
    def probability(self, x) -> float:
        return self.weights[x]
    def fprobability(self, x) -> float:
        return sum([self.probability(xi) for xi in range(x + 1)])
    def graph(self, ax: plt.Axes) -> None:
        ax.bar([x for x in range(len(self.weights))], self.weights)
    def fgraph(self, ax: plt.Axes) -> None:
        x = [x for x in range(len(self.weights))]
        ax.bar(x, [self.fprobability(xi) for xi in x])

class FDiscreteDist:
    def __init__(self, p_func: Callable[[int], float], f_func: Callable[[int], float]):
        self.p_func = p_func
        self.f_func = f_func
    def probability(self, x) -> float:
        return self.p_func(x)
    def fprobability(self, x) -> float:
        return self.f_func(x)
    def graph(self, ax: plt.Axes, x_min: int, x_max: int) -> None:
        x = list(range(x_min, x_max + 1))
        ax.bar(x, [self.p_func(xi) for xi in x])
    def fgraph(self, ax: plt.Axes, x_min: int, x_max: int) -> None:
        x = list(range(x_min, x_max + 1))
        ax.bar(x, [self.f_func(xi) for xi in x])

class BinomialDist(DiscreteDist):
    def __init__(self, n: int, p: float) -> None:
        super().__init__([math.comb(n, x)*p**x*(1 - p)**(n - x) for x in range(n + 1)])
        self.n = n
        self.p = p
    def expected(self):
        return self.n*self.p
    def variance(self):
        return self.n*self.p*(1 - self.p)

class PoissonDist(FDiscreteDist):
    def __init__(self, mu: float) -> None:
        super().__init__(lambda x: math.e**(-mu)*mu**x/math.factorial(x),
                         lambda x: math.e**(-mu)*sum([mu**xi/math.factorial(xi) for xi in range(x + 1)]))
        self.mu = mu
    def expected(self):
        return self.mu
    def variance(self):
        return self.mu

class GeometricDist(DiscreteDist):
    pass

class ContinuousDist:
    def __init__(self, pdf: Callable[[float], float], cdf: Callable[[float], float]) -> None:
        self.pdf = pdf
        self.cdf = cdf
    def probability(self, leq: float, geq: float | None = None) -> float:
        if geq is None:
            return self.cdf(leq)
        else:
            return self.cdf(leq) - self.cdf(geq)
    def graph(self, ax: plt.Axes, x_min: float, x_max: float, ticks: int = 100) -> None:
        x = [x_min + (x_max - x_min)*x/ticks for x in range(ticks)]
        ax.plot(x, [self.pdf(xi) for xi in x])
    def fgraph(self, ax: plt.Axes, x_min: float, x_max: float, ticks: int = 100) -> None:
        x = [x_min + (x_max - x_min)*x/ticks for x in range(ticks)]
        ax.plot(x, [self.cdf(xi) for xi in x])
        
class ExponentialDist(ContinuousDist):
    def __init__(self, mu: float) -> None:
        super().__init__(lambda x: math.e**(-x/mu)/mu, lambda x: 1 - math.e**(-x/mu))
        self.mu = mu
    def graph(self, ax, ticks = 100):
        return super().graph(ax, 0, 4*self.mu, ticks)
    def fgraph(self, ax, ticks = 100):
        return super().fgraph(ax, 0, 4*self.mu, ticks)
    def expected(self) -> float:
        return self.mu
    def variance(self) -> float:
        return self.mu**2
    def standard_dev(self) -> float:
        return self.mu

class NormalDist(ContinuousDist):
    def __init__(self, mu: float, s: float):
        super().__init__(lambda x: math.e**(-(x-mu)**2/(2*s**2))/math.sqrt(2*math.pi*s**2),
                         lambda x: 0.5*(1 + math.erf((x - mu)/(s*math.sqrt(2)))))
        self.mu = mu
        self.s = s
    def graph(self, ax, ticks = 100):
        return super().graph(ax, self.mu - 4*self.s, self.mu + 4*self.s, ticks)
    def fgraph(self, ax, ticks = 100):
        return super().fgraph(ax, self.mu - 4*self.s, self.mu + 4*self.s, ticks)
    def expected(self) -> float:
        return self.mu
    def variance(self) -> float:
        return self.s**2
    def standard_dev(self) -> float:
        return self.s
    def __add__(self, other: "NormalDist") -> "NormalDist":
        return NormalDist(self.mu + other.mu, math.sqrt(self.s**2 + other.s**2))
    
class ZDist(NormalDist):
    def __init__(self):
        super().__init__(0, 1)

class TDist(ContinuousDist):
    def __init__(self, df: int):
        super().__init__(lambda x: math.gamma((df + 1)/2)/(math.sqrt(math.pi*df)*math.gamma(df/2))*(1 + x**2/df)**(-(df + 1)/2),
                         lambda x: 0.5 + x*math.gamma((df + 1)/2)*hyp2f1(0.5, (df + 1)/2, 1.5, -(x**2/df))/(math.sqrt(math.pi*df)*math.gamma(df/2)))
        self.df = df
    def graph(self, ax, ticks = 100):
        return super().graph(ax, -4, 4, ticks)
    def fgraph(self, ax, ticks = 100):
        return super().fgraph(ax, -4, 4, ticks)