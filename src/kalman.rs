#![allow(non_snake_case)]

use nalgebra::{ComplexField, SMatrix, Scalar, SimdValue};

struct VecMat<const D: usize, F: Scalar + SimdValue + ComplexField + Copy> {
    x: SMatrix<F, D, 1>,
    P: SMatrix<F, D, D>,
}

/// Linear state-space `D`-dimensional Kalman filter implementation utilizing the `nalgebra` library.
pub struct KalmanFilter<const D: usize, F: Scalar + SimdValue + ComplexField + Copy> {
    // Model propagation matrix
    F: SMatrix<F, D, D>,

    // Model noise covariance matrix
    Q: SMatrix<F, D, D>,

    // a priori state vector and covariance matrix
    prio: VecMat<D, F>,

    // a posteriori state vector and covariance matrix
    post: Option<VecMat<D, F>>,

    // Previous control input
    uv: Option<SMatrix<F, D, 1>>,
}

impl<const D: usize, F: Scalar + SimdValue + ComplexField + Copy> KalmanFilter<D, F> {
    /// Provide kalman filter with all initial values
    pub fn new(
        F: SMatrix<F, D, D>,
        Q: SMatrix<F, D, D>,
        x_init: SMatrix<F, D, 1>,
        P_init: SMatrix<F, D, D>,
    ) -> Self {
        Self {
            F,
            Q,
            prio: VecMat {
                x: x_init,
                P: P_init,
            },
            post: None,
            uv: None,
        }
    }

    /// Predict new state
    pub fn predict(&mut self) {
        match self.post.as_mut() {
            // Simple prediction, no new observations
            None => {
                self.prio.x = self.F * self.prio.x;
                self.prio.P = self.F * self.prio.P * self.F.transpose() + self.Q;
            }

            // Prediction based on new observations
            Some(post) => {
                // Finish calc for P_post and symmetrize
                post.P = post.P * self.prio.P;

                // Symmetrize
                post.P = (post.P + post.P.transpose()).scale(nalgebra::convert(0.5));

                // Update priors
                self.prio.x = self.F * post.x;
                self.prio.P = self.F * post.P * self.F.transpose() + self.Q;

                // Set posteriors to none
                self.post = None;
            }
        }
    }

    pub fn control_input(&mut self, u: Option<&SMatrix<F, D, 1>>) {
        // Apply input vector
        if let Some(prev_uv) = self.uv {
            self.prio.x -= prev_uv;
        }
        if let Some(uv) = u {
            self.prio.x += uv;
            self.uv = Some(*uv);
        } else {
            self.uv = None
        }
    }

    /// Update filter with new measurements
    pub fn update<const M: usize>(
        &mut self,
        H: &SMatrix<F, M, D>,
        R: &SMatrix<F, M, M>,
        Z: &SMatrix<F, M, 1>,
    ) {
        // Innovation or measurement pre-fit residual
        let y = Z - H * self.prio.x;

        // Innovation (or pre-fit residual) covariance
        let S = H * self.prio.P * H.transpose() + R;

        // Optimal Kalman gain
        let K = self.prio.P * H.transpose() * S.try_inverse().unwrap();

        // Updated (a posteriori) estimate covariance
        self.post = Some(match self.post.as_mut() {
            Some(post) => VecMat {
                x: post.x + K * y,
                P: post.P - K * H,
            },
            None => VecMat {
                x: self.prio.x + K * y,
                P: SMatrix::identity() - K * H,
            },
        });
    }

    /// Get a priori state vector
    #[inline]
    pub fn get_state(&self) -> SMatrix<F, D, 1> {
        if let Some(post) = &self.post {
            post.x
        } else {
            self.prio.x
        }
    }
}
