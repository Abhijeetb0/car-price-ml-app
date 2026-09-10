import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

function App() {

  const [form, setForm] = useState({
    name: "Hyundai Creta 1.6 VTVT S",
    year: 2015,
    km_driven: 25000,
    fuel: "Petrol",
    seller_type: "Individual",
    transmission: "Manual",
    owner: "First Owner",
  });


  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  function handleChange(event) {

    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  }


  async function handleSubmit(event) {

    event.preventDefault();

    setLoading(true);
    setError("");
    setPrediction(null);


    try {

      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            name: form.name,
            year: Number(form.year),
            km_driven: Number(form.km_driven),
            fuel: form.fuel,
            seller_type: form.seller_type,
            transmission: form.transmission,
            owner: form.owner,
          }),
        }
      );


      if (!response.ok) {
        throw new Error("Prediction failed");
      }


      const data = await response.json();

      setPrediction(data.predicted_price);

    } catch (error) {

      setError(
        "Prediction API se connection nahi ho paaya."
      );

    } finally {

      setLoading(false);
    }
  }


  return (

    <main className="app">

      <section className="card">

        <div className="header">

          <p className="eyebrow">
            MACHINE LEARNING
          </p>

          <h1>
            Used Car Price
          </h1>

          <p className="subtitle">
            Enter car details and get an estimated
            resale price from our ML model.
          </p>

        </div>


        <form onSubmit={handleSubmit}>

          <label>
            Car Model

            <input
              type="text"
              name="name"
              value={form.name}
              onChange={handleChange}
              placeholder="e.g. Hyundai Creta 1.6 VTVT S"
              required
            />

          </label>


          <div className="grid">

            <label>
              Manufacturing Year

              <input
                type="number"
                name="year"
                value={form.year}
                onChange={handleChange}
                min="1990"
                max="2026"
                required
              />

            </label>


            <label>
              Kilometers Driven

              <input
                type="number"
                name="km_driven"
                value={form.km_driven}
                onChange={handleChange}
                min="0"
                required
              />

            </label>

          </div>


          <div className="grid">

            <label>
              Fuel

              <select
                name="fuel"
                value={form.fuel}
                onChange={handleChange}
              >

                <option>Petrol</option>
                <option>Diesel</option>
                <option>CNG</option>
                <option>LPG</option>
                <option>Electric</option>

              </select>

            </label>


            <label>
              Transmission

              <select
                name="transmission"
                value={form.transmission}
                onChange={handleChange}
              >

                <option>Manual</option>
                <option>Automatic</option>

              </select>

            </label>

          </div>


          <div className="grid">

            <label>
              Seller Type

              <select
                name="seller_type"
                value={form.seller_type}
                onChange={handleChange}
              >

                <option>Individual</option>
                <option>Dealer</option>
                <option>Trustmark Dealer</option>

              </select>

            </label>


            <label>
              Owner

              <select
                name="owner"
                value={form.owner}
                onChange={handleChange}
              >

                <option>First Owner</option>
                <option>Second Owner</option>
                <option>Third Owner</option>
                <option>Fourth & Above Owner</option>
                <option>Test Drive Car</option>

              </select>

            </label>

          </div>


          <button
            type="submit"
            disabled={loading}
          >

            {loading
              ? "Predicting..."
              : "Predict Car Price"}

          </button>

        </form>


        {error && (

          <div className="error">
            {error}
          </div>

        )}


        {prediction !== null && (

          <div className="result">

            <p>
              Estimated Selling Price
            </p>

            <h2>
              ₹{Number(prediction).toLocaleString("en-IN")}
            </h2>

            <span>
              Prediction generated by the trained ML model
            </span>

          </div>

        )}

      </section>

    </main>
  );
}


export default App;