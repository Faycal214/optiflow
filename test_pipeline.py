from models import get_model_config
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

config = get_model_config("svc")
search_space = config.build_search_space()
wrapper = config.get_wrapper()

# Sample parameters
params = search_space.sample()
print("Sampled params:", params)

# Train and test model
model = wrapper.instantiate(params)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Test score:", score)
