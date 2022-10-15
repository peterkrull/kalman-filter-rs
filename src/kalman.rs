#![allow(unused, non_snake_case)]

use core::ops::Div;

use nalgebra::{self, ArrayStorage, Const, Matrix, SMatrix};


/// Linear state-space Kalman filter implementation utilizing the `nalgebra` library.
pub struct KalmanFilter<const D: usize> {

    // Model propagation matrix
    F: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // Model noise covariance matrix
    Q: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // Model noise covariance matrix
    R: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // Priori estimated state vector
    x_pre: Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>,

    // Priori estimate covaraince
    P_pre: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // Posteriori estimated state vector
    x_post: Option<Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>>,

    // Posteriory estimate covaraince
    P_post: Option<Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>>,
}

impl<const D: usize> KalmanFilter<D> {
    // Provide kalman filter with all initial values
    pub fn new(
        F: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
        Q: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
        R: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
        x_init: Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>,
        P_init: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
    ) -> Self {
        Self {
            F,
            Q,
            R,
            x_pre: x_init,
            P_pre: P_init,
            x_post: Some(x_init),
            P_post: Some(P_init),
        }
    }

    pub fn predict(&mut self, u: Option<Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>>) {
        match (self.x_post.as_mut(), self.P_post.as_mut()) {
            // Simple prediction, no new observations
            (None, None) => {
                self.x_pre = self.F * self.x_pre;
                self.P_pre = self.F * self.P_pre * self.F.transpose() + self.Q;
            }

            // Prediction based on new measurements
            (Some(x_post), Some(P_post)) => {
                // Finish calc for P_post and symmetrize
                *P_post = *P_post * self.P_pre;
                *P_post = (*P_post + (*P_post).transpose()).div(2.0);

                // Update priors
                match u {
                    Some(uv) => {
                        self.x_pre = self.F * *x_post + uv;
                    }
                    None => {
                        self.x_pre = self.F * *x_post;
                    }
                }
                self.P_pre = self.F * *P_post * self.F.transpose() + self.Q;

                // Set posteriors to none
                self.x_post = None;
                self.P_post = None;
            }

            // These cases should not be possible
            (None, Some(_)) => panic!(),
            (Some(_), None) => panic!(),
        }
    }

    pub fn update<const D2: usize>(
        &mut self,
        H: Matrix<f32, Const<D2>, Const<D>, ArrayStorage<f32, D2, D>>,
        R: Matrix<f32, Const<D2>, Const<D2>, ArrayStorage<f32, D2, D2>>,
        Z: Matrix<f32, Const<D2>, Const<1>, ArrayStorage<f32, D2, 1>>,
    ) {
        // Innovation or measurement pre-fit residual
        let y = Z - (H * self.x_pre);

        // Innovation (or pre-fit residual) covariance
        let S = (H * self.P_pre * H.transpose()) + R;

        // Optimal Kalman gain
        let K = self.P_pre * H.transpose() * S.try_inverse().unwrap();

        // Updated (a posteriori) state estimate
        self.x_post = match self.x_post {
            Some(x_post) => Some(x_post + K * y),
            None => Some(self.x_pre + K * y),
        };

        // Updated (a posteriori) estimate covariance
        self.P_post = match self.P_post {
            Some(P_post) => Some(P_post - K * H),
            None => Some(SMatrix::<f32, D, D>::identity() - K * H),
        };

        // Measurement post-fit residual
        let y_post = Z - H * self.x_post.unwrap();
    }
}
