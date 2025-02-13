import math
import matplotlib.pyplot as plt
from scipy.special import hyp2f1
from typing import Callable, Literal



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

def expected(elements: list[int | float]) -> float:
    return mean(elements=elements)
def variance(elements: list[int | float], v_type: Literal["pop", "samp"] = "pop") -> float:
    m = mean(elements=elements)
    n = len(elements)
    if v_type == "pop":
        return sum([(x - m)**2 for x in elements])/n
    else:
        return sum([(x - m)**2 for x in elements])/(n - 1)
def standard_dev(elements: list[int | float], s_type: Literal["pop", "samp"] = "pop") -> float:
    return math.sqrt(variance(elements=elements, v_type=s_type))
def mad(elements: list[int | float]) -> float:
    m = mean(elements=elements)
    n = len(elements)
    return sum([abs(x - m) for x in elements])/n


class DiscreteDist:
    def __init__(self, weights: list[int | float]) -> None:
        total_weight = sum(weights)
        self._weights = [weight/total_weight for weight in weights]
        self._e_x = sum([x*p for x, p in enumerate(self.weights)])
        self._v_x = sum([(x - self.expected())**2*p for x, p in enumerate(self.weights)])
        self._s_x = math.sqrt(self._v_x)

    @property
    def weights(self):
        return self._weights
    @property
    def e_x(self) -> float:
        return self._e_x
    @property
    def v_x(self) -> float:
        return self._v_x
    @property
    def s_x(self) -> float:
        return self._s_x

    def expected(self) -> float:
        return self.e_x
    def variance(self) -> float:
        return self.v_x
    def standard_dev(self) -> float:
        return self.s_x
    def probability(self, x) -> float:
        return self.weights[x]
    def cumulative_probability(self, x) -> float:
        return sum([self.probability(xi) for xi in range(x + 1)])
    def probability_graph(self, ax: plt.Axes) -> None:
        ax.bar(list(range(len(self.weights))), self.weights)
    def cumulative_graph(self, ax: plt.Axes) -> None:
        x = list(range(len(self.weights)))
        ax.bar(x, [self.cumulative_probability(xi) for xi in x])

class FDiscreteDist(DiscreteDist):
    def __init__(self, pdf: Callable[[int], float], *, n: int | None = None,
                 exp: float | None = None, var: float | None = None):
        self._pdf = pdf
        self._n = n
        if exp is None:
            if n is None:
                lim = 1000000
            else:
                lim = n + 1
            exp_sum = 0
            for x in range(lim):
                exp_sum += x*pdf(x)
            self._e_x = exp_sum/(lim)
        else:
            self._e_x = exp
        if var is None:
            if n is None:
                lim = 1000000
            else:
                lim = n + 1
            var_sum = 0
            for x in range(lim):
                var_sum += (x - self._e_x)**2
            self._e_x = var_sum/(lim)
        else:
            self._v_x = var
        self._s_x = math.sqrt(self.v_x)

    @property
    def pdf(self) -> Callable[[int], float]:
        return self._pdf
    
    def probability(self, x) -> float:
        return self.pdf(x)
    def fprobability(self, x) -> float:
        return sum([self.probability(xi) for xi in range(x + 1)])
    def graph(self, ax: plt.Axes, x_min: int, x_max: int) -> None:
        x = list(range(x_min, x_max + 1))
        ax.bar(x, [self.probability(xi) for xi in x])
    def fgraph(self, ax: plt.Axes, x_min: int, x_max: int) -> None:
        x = list(range(x_min, x_max + 1))
        ax.bar(x, [self.fprobability(xi) for xi in x])

class BinomialDist(FDiscreteDist):
    def __init__(self, n: int, p: float) -> None:
        super().__init__(lambda x: math.comb(n, x)*p**x*(1 - p)**(n - x), n=n, exp=n*p, var=n*p*(1 - p))
        self._p = p

    @property
    def p(self) -> float:
        return self._p

class PoissonDist(FDiscreteDist):
    def __init__(self, mu: float) -> None:
        super().__init__(lambda x: math.e**(-mu)*mu**x/math.factorial(x))
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