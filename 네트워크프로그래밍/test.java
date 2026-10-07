class AdvancedOrderService {
    private OrderRepository repository =
        new OrderRepository();

    public int processOrder(
        String orderId,
        int amount,
        PaymentStrategy payment
    ) {
        if (amount <= 0) {
            throw new IllegalArgumentException();
        }

        int payAmount =
            amount >= 100000
                ? (int) (amount * 0.95)
                : amount;

        payment.pay(payAmount);
        repository.saveOrder(
            orderId,
            payAmount
        );

        return payAmount;
    }
}