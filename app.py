from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Cluster names
cluster_names = [
    "Older customers with average income and spending.",
    "Young adults with average income and average spending.",
    "Middle-aged high-income customers with low spending.",
    "Middle-aged high-income customers with high spending.",
    "Young low-income customers with high spending.",
    "Older low-income customers with low spending."
]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Getting input values from the form
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        spending_score = float(request.form['score'])

        # Predicting the cluster
        input_data = [[age, salary, spending_score]]
        cluster = kmeans.predict(input_data)[0]

        # Get the cluster name
        cluster_name = cluster_names[cluster]

        return render_template('index.html', cluster_name=cluster_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
