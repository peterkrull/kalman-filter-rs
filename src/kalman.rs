#![allow(non_snake_case)]

use core::ops::Div;

use nalgebra::{self, ArrayStorage, Const, Matrix, SMatrix};


/// Linear state-space Kalman filter implementation utilizing the `nalgebra` library.
pub struct KalmanFilter<const D: usize> {

    // Model propagation matrix
    F: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // Model noise covariance matrix
    Q: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,

    // a priori state vector and covariance matrix
    prio : VectorMatrix<D>,

    // a posteriori state vector and covariance matrix
    post : Option<VectorMatrix<D>>,
}

struct VectorMatrix<const D : usize> {
    x: Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>,
    P: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>, 
}

impl<const D: usize> KalmanFilter<D> {
    // Provide kalman filter with all initial values
    pub fn new(
        F: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
        Q: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
        x_init: Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>,
        P_init: Matrix<f32, Const<D>, Const<D>, ArrayStorage<f32, D, D>>,
    ) -> Self {
        Self {
            F,
            Q,
            prio: VectorMatrix{ x : x_init, P : P_init },
            post: None,
        }
    }

    pub fn predict(&mut self, u: Option<Matrix<f32, Const<D>, Const<1>, ArrayStorage<f32, D, 1>>>) {
        
        match self.post.as_mut() {
            // Simple prediction, no new observations
            None => {
                self.prio.x = self.F * self.prio.x;
                self.prio.P = self.F * self.prio.P * self.F.transpose() + self.Q;
            }

            // Prediction based on new measurements
            Some(post) => {
                // Finish calc for P_post and symmetrize
                (*post).P = (*post).P * self.prio.P;

                // Symmetrize P
                (*post).P = ((*post).P + ((*post).P).transpose()).div(2.0);

                // Update priors
                if let Some(u) = u {
                    self.prio.x = self.F * (*post).x + u;
                } else {
                    self.prio.x = self.F * (*post).x;
                }

                self.prio.P = self.F * (*post).P * self.F.transpose() + self.Q;

                // Set posteriors to none
                self.post = None;
            }
        }
    }

    pub fn update<const D2: usize>(
        &mut self,
        H: &Matrix<f32, Const<D2>, Const<D>, ArrayStorage<f32, D2, D>>,
        R: &Matrix<f32, Const<D2>, Const<D2>, ArrayStorage<f32, D2, D2>>,
        Z: Matrix<f32, Const<D2>, Const<1>, ArrayStorage<f32, D2, 1>>,
    ) {
        // Innovation or measurement pre-fit residual
        let y = Z - (H * self.prio.x);

        // Innovation (or pre-fit residual) covariance
        let S = (H * self.prio.P * H.transpose()) + R;

        // Optimal Kalman gain
        let K = self.prio.P * H.transpose() * S.try_inverse().unwrap();

        // Updated (a posteriori) estimate covariance
        self.post = Some(match self.post.as_mut() {
            Some(post) => {
                VectorMatrix {
                    x : post.x + K * y,
                    P : post.P - K * H,
                }
            },
            None => {
                VectorMatrix {
                    x : self.prio.x + K * y,
                    P : SMatrix::<f32, D, D>::identity() - K * H,
                }
            }
        });

        // Measurement post-fit residual
        // let y_post = Z - H * self.x_post.unwrap();
    }
}
