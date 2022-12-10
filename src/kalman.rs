#![allow(non_snake_case)]

use nalgebra::SMatrix;

struct VecMat<const D : usize> {
    x: SMatrix<f32,D,1>,
    P: SMatrix<f32,D,D>, 
}

/// Linear state-space `D`-dimensional Kalman filter implementation utilizing the `nalgebra` library.
pub struct KalmanFilter<const D: usize> {

    // Model propagation matrix
    F: SMatrix<f32,D,D>,

    // Model noise covariance matrix
    Q: SMatrix<f32,D,D>,

    // a priori state vector and covariance matrix
    prio : VecMat<D>,

    // a posteriori state vector and covariance matrix
    post : Option<VecMat<D>>,

}

impl<const D: usize> KalmanFilter<D> {
    
    /// Provide kalman filter with all initial values
    pub fn new( F: SMatrix<f32,D,D>, Q: SMatrix<f32,D,D>, x_init: SMatrix<f32,D,1>, P_init: SMatrix<f32,D,D>) -> Self {
        Self { F,Q,prio: VecMat{ x : x_init, P : P_init },post: None }
    }

    /// Predict new state
    pub fn predict(&mut self, u: Option<&SMatrix<f32,D,1>>) {

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

                // Symmetrize P
                post.P = (post.P + post.P.transpose())/2.0;

                // Update priors
                self.prio.x = self.F * post.x;
                self.prio.P = self.F * post.P * self.F.transpose() + self.Q;

                // Set posteriors to none
                self.post = None;
            }

        }

        // Apply input vector
        if let Some(uv) = u {
            self.prio.x += uv;
        }
    }

    /// Update filter with new measurements
    pub fn update <const M: usize>( &mut self, H: &SMatrix<f32,M,D>, R: &SMatrix<f32,M,M>, Z: &SMatrix<f32,M,1>) {
        
        // Innovation or measurement pre-fit residual
        let y = Z - H * self.prio.x;

        // Innovation (or pre-fit residual) covariance
        let S = H * self.prio.P * H.transpose() + R;

        // Optimal Kalman gain
        let K = self.prio.P * H.transpose() * S.try_inverse().unwrap();

        // Updated (a posteriori) estimate covariance
        self.post = Some(match self.post.as_mut() {
            Some(post) => {
                VecMat {
                    x : post.x + K * y,
                    P : post.P - K * H,
                }
            },
            None => {
                VecMat {
                    x : self.prio.x + K * y,
                    P : SMatrix::identity() - K * H,
                }
            }
        });
    }

    /// Get a priori state vector
    #[inline]
    pub fn get_state (&self) -> SMatrix<f32,D,1> {
        self.prio.x
    }
}
