#![allow(non_snake_case)]

use nalgebra::{ComplexField, SMatrix, Scalar, SimdValue};

struct VecMat<const N: usize, F: Scalar + SimdValue + ComplexField + Copy> {
    x: SMatrix<F, N, 1>,
    P: SMatrix<F, N, N>,
}

/// Linear state-space `D`-dimensional Kalman filter implementation utilizing the `nalgebra` library.
pub struct KalmanFilter<const Nx: usize, const Nu: usize, F: Scalar + SimdValue + ComplexField + Copy> {

    // Model propagation matrix
    A: SMatrix<F, Nx, Nx>,

    // Model propagation matrix
    B: SMatrix<F, Nx, Nu>,

    // Model noise covariance matrix
    Q: SMatrix<F, Nx, Nx>,

    // a priori state vector and covariance matrix
    prio: VecMat<Nx, F>,

    // a posteriori state vector and covariance matrix
    post: Option<VecMat<Nx, F>>,

}

impl<const Nx: usize, const Nu: usize, F: Scalar + SimdValue + ComplexField + Copy> KalmanFilter<Nx, Nu, F> {
    /// Provide kalman filter with all initial values
    pub fn new(
        A: SMatrix<F, Nx, Nx>,
        B: Option<SMatrix<F, Nx, Nu>>,
        Q: SMatrix<F, Nx, Nx>,
        x_init: SMatrix<F, Nx, 1>,
        P_init: SMatrix<F, Nx, Nx>,
    ) -> Self {
        Self {
            A,
            B : match B {
                Some(B) => B,
                None => SMatrix::from_element(nalgebra::convert(0.0)),
            },
            Q,
            prio: VecMat {
                x: x_init,
                P: P_init,
            },
            post: None,
        }
    }

    pub fn set_A(&mut self, new_A : SMatrix<F, Nx, Nx>) {
        self.A = new_A;
    }

    pub fn set_B(&mut self, new_B : Option<SMatrix<F, Nx, Nu>>) {
        self.B = match new_B {
            Some(B) => B,
            None => SMatrix::from_element(nalgebra::convert(0.0)),
        };
    }

    /// Predict new state
    pub fn predict(&mut self) {
        let u : SMatrix<F, Nu, 1> = SMatrix::from_element(nalgebra::convert(0.0));
        self.predict_with_input(u)
    }

    /// Predict new state using input
    pub fn predict_with_input(&mut self, u : SMatrix<F, Nu, 1>) {
        match self.post.as_mut() {
            // Simple prediction, no new observations
            None => {
                self.prio.x = self.A * self.prio.x + self.B*u;
                self.prio.P = self.A * self.prio.P * self.A.transpose() + self.Q;
            }

            // Prediction based on new observations
            Some(post) => {
                // Finish calc for P_post and symmetrize
                post.P = post.P * self.prio.P;

                // Symmetrize
                post.P = (post.P + post.P.transpose()).scale(nalgebra::convert(0.5));

                // Update priors
                self.prio.x = self.A * post.x + self.B*u;
                self.prio.P = self.A * post.P * self.A.transpose() + self.Q;

                // Set posteriors to none
                self.post = None;
            }
        }
    }

    /// Update filter with new measurements
    pub fn update<const Ny: usize>(
        &mut self,
        C: &SMatrix<F, Ny, Nx>, // Output matrix
        R: &SMatrix<F, Ny, Ny>, // Covariance
        y: &SMatrix<F, Ny, 1>, // Measurement
    ) {
        // Measurement prediction residual
        let y_res = y - C * self.prio.x;

        // Innovation (or pre-fit residual) covariance
        let S = C * self.prio.P * C.transpose() + R;

        // Optimal Kalman gain
        let Some(Sinv) = S.try_inverse() else { return };
        let K = self.prio.P * C.transpose() * Sinv;

        // Updated (a posteriori) estimate covariance
        self.post = Some(match self.post.as_mut() {
            Some(post) => VecMat {
                x: post.x + K * y_res,
                P: post.P - K * C,
            },
            None => VecMat {
                x: self.prio.x + K * y_res,
                P: SMatrix::identity() - K * C,
            },
        });
    }

    /// Get state vector
    #[inline]
    pub fn get_state(&self) -> SMatrix<F, Nx, 1> {
        if let Some(post) = &self.post {
            post.x
        } else {
            self.prio.x
        }
    }
}
