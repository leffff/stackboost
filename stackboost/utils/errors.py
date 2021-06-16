class NotFittedError(Exception):
    def __init__(self, message="This StackBoost model has not been fitted yet. Use .fit() to train the model."):
        self.message = message
        super().__init__(self.message)
