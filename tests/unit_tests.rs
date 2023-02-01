extern crate std;

#[cfg(test)]
mod tests {

    use assert_approx_eq::assert_approx_eq;
    use kalman_filter::kalman::KalmanFilter;
    use nalgebra::matrix;

    #[test]
    fn gravity_fall() {
        //Initialize filter
        const TD: f32 = 1. / 100.;
        let mut filter = KalmanFilter::new(
            matrix![
                1., TD, 0.5*TD.powf(2.0) ;
                0., 1., TD ;
                0., 0., 1. ],
            matrix![
                1.,0.,0.;
                0.,1.,0.;
                0.,0.,1.],
            matrix![
                0.;0.;0.;],
            matrix![
                1.,0.,0.;
                0.,1.,0.;
                0.,0.,1.],
        );

        // Simulate with an external input as the
        // gravitational acceleration
        const G: f32 = 9.82;
        let seconds: usize = 5;
        for _ in 0..100 * seconds {
            filter.control_input(Some(&matrix![0.;0.;G]));
            filter.predict();
        }

        let state = filter.get_state();

        // Expected states
        let pos = G * 0.5 * (seconds as f32).powf(2.0);
        let vel = G * seconds as f32;
        let acc = G;

        assert_approx_eq!(state[0], pos, 1e-3);
        assert_approx_eq!(state[1], vel, 1e-3);
        assert_approx_eq!(state[2], acc, 1e-3);
    }
}
