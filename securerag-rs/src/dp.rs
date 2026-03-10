pub struct RDPAccountant {
    pub epsilon_max: f64,
    pub delta: f64,
    pub spent: f64,
    pub orders: Vec<f64>,
    pub rdp_epsilons: Vec<f64>,
}

impl RDPAccountant {
    pub fn new(epsilon_max: f64, delta: f64) -> Self {
        let orders = vec![2.0, 4.0, 8.0, 16.0, 32.0];
        let rdp_epsilons = vec![0.0; orders.len()];
        Self {
            epsilon_max,
            delta,
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
            .zip(self.orders.iter())
            .filter(|(_, alpha)| **alpha > 1.0)
            .map(|(rdp, alpha)| {
                let delta_term = (1.0 / self.delta).ln() / (alpha - 1.0);
                *rdp + delta_term
            })
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::RDPAccountant;

    #[test]
    fn rdp_to_dp_uses_delta_conversion_term() {
        let mut acc = RDPAccountant::new(1_000.0, 1e-5);
        acc.consume_rdp(1.0).expect("consume should succeed");

        let expected = acc
            .orders
            .iter()
            .zip(acc.rdp_epsilons.iter())
            .map(|(alpha, rdp)| rdp + (1.0_f64 / 1e-5_f64).ln() / (alpha - 1.0))
            .fold(f64::INFINITY, f64::min);

        assert!((acc.spent - expected).abs() < 1e-9);
    }

    #[test]
    fn consume_respects_budget_after_conversion() {
        let mut acc = RDPAccountant::new(0.5, 1e-5);
        let err = acc.consume_rdp(10.0).expect_err("budget should be exhausted");
        assert!(err.contains("Budget exhausted"));
    }
}
