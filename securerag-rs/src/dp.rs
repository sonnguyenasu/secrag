pub struct RDPAccountant {
    pub epsilon_max: f64,
    pub spent: f64,
    pub orders: Vec<f64>,
    pub rdp_epsilons: Vec<f64>,
}

impl RDPAccountant {
    pub fn new(epsilon_max: f64) -> Self {
        let orders = vec![2.0, 4.0, 8.0, 16.0, 32.0];
        let rdp_epsilons = vec![0.0; orders.len()];
        Self {
            epsilon_max,
            spent: 0.0,
            orders,
            rdp_epsilons,
        }
    }

    pub fn consume_rdp(&mut self, sigma: f64) -> Result<(), String> {
        if sigma <= 0.0 {
            return Err("sigma must be > 0".to_string());
        }
        for (rdp, order) in self.rdp_epsilons.iter_mut().zip(self.orders.iter()) {
            *rdp += order / (2.0 * sigma * sigma);
        }
        self.spent = self.rdp_to_dp();
        if self.spent > self.epsilon_max {
            return Err(format!(
                "Budget exhausted: {:.3} / {:.3}",
                self.spent, self.epsilon_max
            ));
        }
        Ok(())
    }

    fn rdp_to_dp(&self) -> f64 {
        self.rdp_epsilons
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
}
