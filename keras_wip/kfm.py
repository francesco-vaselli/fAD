import keras
import numpy as np
from keras import ops


class FMWrapper(keras.Model):
    def __init__(self, base_model: keras.Model):
        super().__init__()
        self.base_model = base_model
        self._seed = keras.random.SeedGenerator()

    def call(self, inputs, training=None, mask=None):
        x1 = inputs
        bsz = ops.shape(x1)[0]
        t = keras.random.uniform((bsz, 1, 1), 0, 1, seed=self._seed)
        x0 = keras.random.normal(x1.shape, seed=self._seed)
        xt = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        v_pred = self.base_model([xt, t], training=True)
        loss = keras.losses.mean_squared_error(v_target, v_pred)
        return loss

    def _sample(self, x, ts, return_trajectories=False):
        dt = ops.diff(ts)
        trajectories = []
        for t, dt in zip(ts[:-1], dt):
            _t = ops.full((len(x), 1, 1), t, dtype=x.dtype)
            v = self.base_model([x, _t])
            x = x + v * dt
            if return_trajectories:
                trajectories.append(ops.convert_to_numpy(x))
        if return_trajectories:
            return np.array(trajectories)
        return x

    def _uniform_sample(self, x, t0, t1, n_steps):
        dt = (t1 - t0) / n_steps
        for step in range(n_steps):
            t = t0 + step * dt
            v = self.base_model([x, ops.reshape(t, (-1, 1, 1))])
            x = x + v * dt
        return x

    def generate(self, n_steps=32, n_samples=1024, batch_size: int | None = None):
        x0 = keras.random.normal((n_samples, 19, 3), seed=self._seed)
        ts = ops.linspace(0, 1, n_steps + 1)
        if batch_size is None or n_samples <= batch_size:
            return self._sample(x0, ts)
        return self.sample(x0, ts, batch_size=batch_size)

    def sample(self, x, ts, batch_size=1024, return_trajectories=False):
        n = len(x)
        n_batch = (n + batch_size - 1) // batch_size
        ret = []
        for i in range(n_batch):
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            ret.append(self._sample(ops.array(x_batch), ops.array(ts), return_trajectories))
        return np.concatenate(ret, axis=0)

    def predict_score(self, x: np.ndarray, batch_size=1024, verbose=0, mode=0) -> np.ndarray:
        t = np.ones((len(x), 1, 1), dtype=x.dtype)
        v = self.base_model.predict([x, t], batch_size=batch_size, verbose=verbose)  # type: ignore
        match mode:
            case 0:
                return v
            case 1:
                return np.sum(v**2, axis=(1, 2))
            case 2:
                return np.sum((x - v)**2, axis=(1, 2))
            case _:
                raise ValueError(f'Unknown mode {mode}')


class FMWrapperReflow(FMWrapper):
    def __init__(self, base_model: keras.Model, teacher: FMWrapper, reflow_steps=10):
        super().__init__(base_model)
        self.teacher = teacher
        self.teacher.trainable = False
        self.reflow_steps = reflow_steps

    def call(self, inputs, training=None, mask=None):

        if not training:
            return super().call(inputs, training=training, mask=mask)

        bsz = len(inputs)

        t = keras.random.uniform((bsz, 1, 1), 0, 1, seed=self._seed)
        x0 = keras.random.normal((bsz, 19, 3), seed=self._seed)
        x1 = self.teacher.generate(n_samples=bsz, n_steps=self.reflow_steps)
        x1 = ops.stop_gradient(x1)
        xt = (1 - t) * x0 + t * x1

        v_target = x1 - x0
        v_pred = self.base_model([xt, t], training=True)
        loss = keras.losses.mean_squared_error(v_target, v_pred)
        return loss
