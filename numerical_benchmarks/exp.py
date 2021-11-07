import numpy as np
np.set_printoptions(precision=4)

def normalize(x):
    return x / (np.linalg.norm(x) + 1e-12)

def set_function(f_type):
    if f_type == 'orig':
        n = 500
        x_OPT = np.arange(n, 0, -1) / (n + 1)
        L1 = 4
        x = np.zeros(n)
        current_row = 1
        base = 2
        tau = 0

        def f(x):
            return 0.5 * x[0] ** 2 + 0.5 * np.sum(np.square((x - np.roll(x, -1))[:-1])) + 0.5 * x[-1] ** 2 - x[0]

        def grad(x):
            grad = np.zeros_like(x)
            grad[0] = 2 * x[0] - x[1] - 1
            grad[1:-1] = 2 * x[1:-1] - x[:-2] - x[2:]
            grad[-1] = 2 * x[-1] - x[-2]
            return grad

        OPT = f(x_OPT)
        S = f(x) - OPT
    elif f_type == 'quad':
        n = 500
        step = 1
        spectrum = (np.arange((n-1) * step + 1, 1-1e-10, -step) / n / step)
        # spectrum = n / (2.0 ** np.arange(n))
        # spectrum = np.ones(n)
        # spectrum[0] = 2560
        # spectrum = np.ones(n) * 256
        # spectrum[-1] = 1
        L1 = 2 * spectrum[0]
        x_OPT = np.zeros(n)
        x = np.zeros(n)
        x[-1] = n
        # x = np.ones(n)
        current_row = 1
        base = 2
        tau = 2 * spectrum[-1]

        def f(x):
            return np.sum(spectrum * x * x)

        def f_inv(y):
            return np.sqrt(y / spectrum)

        def grad(x):
            return 2 * spectrum * x

        OPT = f(x_OPT)
        S = f(x) - OPT
    elif f_type == 'quad_area':
        n = 500
        x_OPT = np.zeros(n)
        x = np.zeros(n)
        x[-1] = 5 * np.sqrt(n)
        current_row = 1
        base = 2
        area_spectrum = np.arange(n, 0, -1) / n
        L1 = area_spectrum[0]
        tau = area_spectrum[-1]

        def f(x):
            rad = np.linalg.norm(x * np.sqrt(area_spectrum))
            if rad <= 1:
                return 0.5 * rad * rad
            else:
                return rad - 0.5

        def grad(x):
            rad = np.linalg.norm(x * np.sqrt(area_spectrum))
            if rad <= 1:
                return area_spectrum * x
            else:
                return area_spectrum * x / rad

        OPT = f(x_OPT)
        S = f(x) - OPT


    return f, grad, x, n, L1, tau, OPT, S, current_row, base

class Avg:
    def __init__(self, direct=False):
        self.list = []
        self.direct = direct

    def update(self, value):
        if self.direct:
            self.list.append(value)
        else:
            self.list.append(value ** 2)

    @property
    def avg(self):
        if self.direct:
            return np.mean(self.list)
        else:
            return np.sqrt(np.mean(self.list))

    def clear(self):
        self.list = []

def cosine_similarity(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))


class RGF:
    def __init__(self, lr, q=1):
        self.svrg = False
        self.lr = lr
        self.q = q
        self.post_sim = Avg()

    def return_update(self, x):
        grad_x = grad(x)
        grad_x /= np.linalg.norm(grad_x)
        prior = np.random.normal(size=n)
        prior = prior / np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(x + mu * u) - f(x)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(x + mu * prior) - f(x)) / mu
        g_mu = g_rand + prior * derivative_prior
        norm_g_mu = np.linalg.norm(g_mu)
        if norm_g_mu == 0:
            g_mu = np.random.normal(size=n)
            g_mu /= np.linalg.norm(g_mu)
            g_mu *= ((f(x + mu * g_mu) - f(x)) / mu)
        self.post_sim.update(np.dot(grad_x, normalize(g_mu)))
        return x - (self.lr * g_mu)

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)

class PRGF:
    def __init__(self, lr, lag=900, lag2=0, q=1):
        self.svrg = False
        self.lr = lr
        self.g_mus = []
        self.grad_xs = []
        self.lag = lag
        self.lag2 = lag2
        self.q = q

        for _ in range(lag):
            g_mu = np.random.normal(size=n)
            g_mu /= np.linalg.norm(g_mu)
            self.g_mus.append(g_mu)
            self.grad_xs.append(g_mu)

        self.prior_sim = Avg()
        self.g_sim = Avg()
        self.post_sim = Avg()
        self.prior_old_sim = Avg()

    def return_update(self, x):
        grad_x = grad(x)
        grad_x /= np.linalg.norm(grad_x)
        u = np.random.normal(size=n)
        prior = self.g_mus[0]
        if self.lag2 > 0:
            prior = prior + self.g_mus[self.lag - self.lag2]
        prior = prior / np.linalg.norm(prior)
        grad_x_prior = self.grad_xs[0]
        self.prior_old_sim.update(np.dot(prior, grad_x_prior))
        self.g_sim.update(np.dot(grad_x, grad_x_prior))
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(x + mu * u) - f(x)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(x + mu * prior) - f(x)) / mu
        self.prior_sim.update(np.dot(prior, grad_x))
        g_mu = g_rand + prior * derivative_prior
        norm_g_mu = np.linalg.norm(g_mu)
        if norm_g_mu == 0:
            g_mu = np.random.normal(size=n)
            g_mu /= np.linalg.norm(g_mu)
            g_mu *= ((f(x + mu * g_mu) - f(x)) / mu)
        self.post_sim.update(np.dot(grad_x, normalize(g_mu)))
        self.g_mus.append(g_mu)
        self.g_mus = self.g_mus[-self.lag:]
        self.grad_xs.append(grad_x)
        self.grad_xs = self.grad_xs[-self.lag:]
        return x - (self.lr * g_mu)

    def clear(self):
        self.prior_sim.clear()
        self.g_sim.clear()
        self.post_sim.clear()
        self.prior_old_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f} g similarity: {:.4f} Prior similarity: {:.4f} Prior old similarity: {:.4f} d: {:.4f}".format(self.post_sim.avg, self.g_sim.avg, self.prior_sim.avg, self.prior_old_sim.avg, self.prior_sim.avg / self.prior_old_sim.avg)


class ARS:
    def __init__(self, L, v_ini, q=1, tau=0):
        self.L = L
        self.ini_gamma = L
        self.gamma = L
        self.v = v_ini
        self.q = q
        self.tau = tau
        self.old_value = 1e10
        self.post_sim = Avg()

    def return_update(self, x):
        theta = (self.q + 1) ** 2 / (n ** 2) / self.L
        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        self.gamma = (1 - alpha) * self.gamma + alpha * self.tau
        y = (1 - beta) * x + beta * self.v

        prior = np.random.normal(size=n)
        prior /= np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        for u in us:
            derivative_rand = (f(y + mu * u) - f(y)) / mu
            g_rand += u * derivative_rand
        derivative_prior = (f(y + mu * prior) - f(y)) / mu
        g_mu = g_rand + prior * derivative_prior
        self.post_sim.update(np.dot(normalize(grad(y)), normalize(g_mu)))
        g1 = g_mu
        g2 = n / (self.q + 1) * g_mu
        lmda = alpha / self.gamma * self.tau
        self.v = (1 - lmda) * self.v + lmda * y - theta / alpha * g2

        if f(y) > self.old_value:
            self.restart(y - 1 / self.L * g1)
        else:
            self.old_value = f(y)

        return y - 1 / self.L * g1

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)

    def restart(self, x):
        self.v = x
        self.gamma = self.ini_gamma
        self.old_value = 1e10
        print("Restarted")


class PARS:
    def __init__(self, L, v_ini, q=1, tau=0):
        self.L = L
        self.ini_gamma = L
        self.gamma = L
        self.v = v_ini
        self.q = q
        self.tau = tau

        self.old_value = 1e10
        self.g_mu = np.random.normal(size=n)
        self.adaptive_theta = 1e-8
        self.post_sim = Avg()

    def return_update(self, x):
        theta = self.adaptive_theta
        k = theta * (self.gamma - self.tau)
        alpha = (-k + np.sqrt(k * k + 4 * theta * self.gamma)) / 2
        beta = alpha * self.gamma / (self.gamma + alpha * self.tau)
        self.gamma = (1 - alpha) * self.gamma + alpha * self.tau
        y = (1 - beta) * x + beta * self.v

        prior = self.g_mu
        prior = prior / np.linalg.norm(prior)
        us = []
        for _ in range(self.q):
            u = np.random.normal(size=n)
            u -= np.dot(u, prior) * prior
            assert np.abs(np.dot(u / np.linalg.norm(u), prior)) <= 1e-4
            for old_u in us:
                u -= np.dot(u, old_u) * old_u
            u /= np.linalg.norm(u)
            us.append(u)
        g_rand = 0
        drs = []
        for u in us:
            derivative_rand = (f(y + mu * u) - f(y)) / mu
            drs.append(derivative_rand)
            g_rand += u * derivative_rand
        derivative_prior = (f(y + mu * prior) - f(y)) / mu
        g_mu = g_rand + prior * derivative_prior

        self.post_sim.update(np.dot(normalize(grad(y)), normalize(g_mu)))
        g1 = g_mu
        self.g_mu = g_mu

        norm_term = np.mean(np.square(drs))
        est_norm2 = norm_term * (n - 1) + derivative_prior ** 2
        Dt = derivative_prior ** 2 / est_norm2
        self.adaptive_theta = (Dt + self.q / (n - 1) * (1 - Dt)) / self.L / (Dt + (n - 1) / self.q * (1 - Dt))

        g_prior = prior * derivative_prior
        g2 = (n - 1) / self.q * g_rand + g_prior

        lmda = alpha / self.gamma * self.tau
        self.v = (1 - lmda) * self.v + lmda * y - (theta / alpha * g2)

        if f(y) > self.old_value:
            self.restart(y - 1 / self.L * g1)
        else:
            self.old_value = f(y)

        return y - 1 / self.L * g1

    def clear(self):
        self.post_sim.clear()

    def statistics(self):
        return "Similarity: {:.4f}".format(self.post_sim.avg)

    def restart(self, x):
        self.v = x
        self.gamma = self.ini_gamma
        self.old_value = 1e10
        print("Restarted")


f_types = ['quad', 'orig', 'quad_area']
optim_types = ['rgf', 'rgf-lr25', 'prgf', 'prgf-lr25', 'prgf-lr50', 'ars', 'ars-lr25', 'pars', 'pars-lr25', 'pars-lr50']

for ite in range(5):
    for f_type in f_types:
        for optim_type in optim_types:
            mu = 1e-6
            f, grad, x, n, L1, tau, OPT, S, current_row, base = set_function(f_type)
            print(ite, f_type, optim_type)
            print("mu={}, OPT={}, S={}, f(x0)-OPT={}".format(mu, OPT, S, f(x) - OPT))
            if f_type == 'quad':
                n_epochs = 200
            elif f_type == 'orig':
                n_epochs = 500
            elif f_type == 'quad_area':
                n_epochs = 200

            q = 10
            if optim_type == 'rgf':
                L1 *= 1
                optim = RGF(lr=1 / L1, q=q)
            elif optim_type == 'rgf-lr25':
                L1 *= 25
                optim = RGF(lr=1 / L1, q=q)
            elif optim_type == 'prgf':
                L1 *= 1
                optim = PRGF(lr=1 / L1, lag=1, q=q)
            elif optim_type == 'prgf-lr25':
                L1 *= 25
                optim = PRGF(lr=1 / L1, lag=1, q=q)
            elif optim_type == 'prgf-lr50':
                L1 *= 50
                optim = PRGF(lr=1 / L1, lag=1, q=q)
            elif optim_type == 'ars':
                L1 *= 1
                optim = ARS(L=L1, v_ini=x, q=q)
            elif optim_type == 'ars-lr25':
                L1 *= 25
                optim = ARS(L=L1, v_ini=x, q=q)
            elif optim_type == 'pars':
                L1 *= 1
                optim = PARS(L=L1, v_ini=x, q=q)
            elif optim_type == 'pars-lr25':
                L1 *= 25
                optim = PARS(L=L1, v_ini=x, q=q)
            elif optim_type == 'pars-lr50':
                L1 *= 50
                optim = PARS(L=L1, v_ini=x, q=q)
            else:
                raise Exception

            errs = []

            for epoch in range(1, n_epochs + 1):
                if hasattr(optim, 'q'):
                    q_per = optim.q + 1
                else:
                    q_per = 2
                for i in range(n // q_per):
                    x = optim.return_update(x)
                print(optim.statistics())
                optim.clear()

                err = (f(x) - OPT) / S
                errs.append(err)
                # print(err)
                while err <= base ** (-current_row):
                    print(current_row, epoch)
                    current_row += 1

            errs = np.array(errs)
            np.save('exp/{}_{}_{}.npy'.format(f_type, optim_type, ite), errs)
