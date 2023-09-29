#[cfg(test)]
mod tests {
    extern crate std;

    use assert_approx_eq::assert_approx_eq;
    use kalman_filter::kalman::KalmanFilter;
    use nalgebra::matrix;
    use rand::random;

    #[test]
    fn gravity_fall_1hz() { gravity_fall_frequency(1) }

    #[test]
    fn gravity_fall_5hz() { gravity_fall_frequency(5) }

    #[test]
    fn gravity_fall_10hz() { gravity_fall_frequency(10) }

    #[test]
    fn gravity_fall_100hz() { gravity_fall_frequency(100) }

    #[test]
    fn gravity_fall_100hz_measurement() {

            for _ in 0..10 {
            let mut mean_pos  = 0.0;
            let mut mean_vel  = 0.0;
            let mut true_pos: f32 = 0.0;
            let mut true_vel: f32 = 0.0;
            let mut est_pos: f32;
            let mut est_vel: f32;
            
            // Average over 10 runs
            for _ in 0..10 {
                ((est_pos,true_pos),(est_vel,true_vel)) = gravity_fall_measurement(100);
                mean_pos += est_pos;
                mean_vel += est_vel;
            }
            mean_pos /= 10.0;
            mean_vel /= 10.0;

            assert_approx_eq!(mean_pos, true_pos, 1.);
            assert_approx_eq!(mean_vel, true_vel, 1.);
        }
    }


    #[test]
    fn gravity_fall_100hz_async_measurement() {

            for _ in 0..10 {
            let mut mean_pos  = 0.0;
            let mut mean_vel  = 0.0;
            let mut true_pos: f32 = 0.0;
            let mut true_vel: f32 = 0.0;
            let mut est_pos: f32;
            let mut est_vel: f32;
            
            // Average over 10 runs
            for _ in 0..10 {
                ((est_pos,true_pos),(est_vel,true_vel)) = gravity_fall_async_measurement(100);
                mean_pos += est_pos;
                mean_vel += est_vel;
            }
            mean_pos /= 10.0;
            mean_vel /= 10.0;

            assert_approx_eq!(mean_pos, true_pos, 1.);
            assert_approx_eq!(mean_vel, true_vel, 1.);
        }
    }

    fn gravity_fall_frequency(hz:usize) {
        //Initialize filter
        let td: f32 = 1. / (hz as f32);
        let mut filter = KalmanFilter::new(
            matrix![
                1., td ;
                0., 1. ],
            Some(matrix![
                1.,0.;
                0.,1.]),
            matrix![
                1.,0.;
                0.,1.],
            matrix![
                0.;0.;],
            matrix![
                1.,0.;
                0.,1.],
        );

        // Simulate with an external input as the gravitational acceleration
        const G: f32 = 9.82;
        let seconds: usize = 5;
        for _ in 0..hz * seconds {
            filter.predict_with_input(matrix![ 0.5*td.powf(2.0)*G ; td*G ]);
        }

        let state = filter.get_state();

        // Expected states
        let pos = G * 0.5 * (seconds as f32).powf(2.0);
        let vel = G * seconds as f32;
        
        assert_approx_eq!(state[0], pos, 1e-3);
        assert_approx_eq!(state[1], vel, 1e-3);

    }


    fn gravity_fall_measurement(hz:usize) -> ((f32,f32),(f32,f32)) {
        //Initialize filter
        let td: f32 = 1. / (hz as f32);
        let mut filter = KalmanFilter::new(
            matrix![
                1., td ;
                0., 1. ],
            Some(matrix![
                1.,0.;
                0.,1.]),
            matrix![
                1.,0.;
                0.,1.],
            matrix![
                0.;0.],
            matrix![
                1.,0.;
                0.,1.],
        );

        const G: f32 = 9.82;
        let seconds: usize = 5;
        for i in 0..hz * seconds {

            // Positional measurement
            if i%10 == 0 {
                let s = i as f32 / hz as f32;
                let p = G * 0.5 * (s as f32).powf(2.0);

                let noise_p = p + (random::<f32>() - 0.5);
    
                filter.update(
                    &matrix![1.,0.], 
                    &matrix![1.],
                    &matrix![noise_p]
                );
            }
    
            // Simulate with an external input as the gravitational acceleration
            filter.predict_with_input(matrix![ 0.5*td.powf(2.0)*G ; td*G ]);
    
        }

        let state = filter.get_state();

        // Expected states
        let pos = G * 0.5 * (seconds as f32).powf(2.0);
        let vel = G * seconds as f32;

        ((state[0],pos),(state[1],vel))
        
    }



    fn gravity_fall_async_measurement(hz:usize) -> ((f32,f32),(f32,f32)) {
        //Initialize filter
        let td: f32 = 1. / (hz as f32);
        let mut filter = KalmanFilter::new(
            matrix![
                1., td ;
                0., 1. ],
            Some(matrix![
                1.,0.;
                0.,1.]),
            matrix![
                1.,0.;
                0.,1.],
            matrix![
                0.;0.],
            matrix![
                1.,0.;
                0.,1.],
        );
        
        const G: f32 = 9.82;
        let seconds: usize = 5;
        for i in 0..hz * seconds {

            // Positional measurement
            if i%20 == 0 {
                let s = i as f32 / hz as f32;
                let pos = G * 0.5 * (s as f32).powf(2.0);

                let pos_noise = pos + (random::<f32>() - 0.5);

                filter.update(
                    &matrix![1.,0.], 
                    &matrix![1.],
                    &matrix![pos_noise]
                );
            }

            // Relatively faster velocity measurement
            if i%5 == 0 {
                let s = i as f32 / hz as f32;
                let vel = G * s as f32;

                let vel_noise = vel + (random::<f32>() - 0.5);
                    
                filter.update(
                    &matrix![0.,1.], 
                    &matrix![1.],
                    &matrix![vel_noise]
                );
            }

            // Simulate with an external input as the gravitational acceleration
            filter.predict_with_input(matrix![ 0.5*td.powf(2.0)*G ; td*G ]);
    
        }

        let state = filter.get_state();

        // Expected states
        let pos = G * 0.5 * (seconds as f32).powf(2.0);
        let vel = G * seconds as f32;

        ((state[0],pos),(state[1],vel))
        
    }

}
