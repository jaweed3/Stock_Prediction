document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predictBtn');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const predictedPriceElement = document.getElementById('predictedPrice');
    const actualPriceElement = document.getElementById('actualPrice');
    const differenceElement = document.getElementById('difference');
    
    predictBtn.addEventListener('click', async function() {
        // Show loading, hide results
        loadingDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        predictBtn.disabled = true;
        
        try {
            // Fetch the latest data from our API
            const response = await fetch('/predict-latest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Format the results
            const predictedPrice = data.predicted_price;
            const actualPrice = data.actual_price;
            const difference = predictedPrice - actualPrice;
            const percentDifference = (difference / actualPrice) * 100;
            
            // Update the UI
            predictedPriceElement.textContent = `$${predictedPrice.toFixed(2)}`;
            actualPriceElement.textContent = `$${actualPrice.toFixed(2)}`;
            
            const differenceText = `$${Math.abs(difference).toFixed(2)} (${Math.abs(percentDifference).toFixed(2)}%)`;
            differenceElement.textContent = difference >= 0 ? `+${differenceText}` : `-${differenceText}`;
            differenceElement.className = difference >= 0 ? 'value positive' : 'value negative';
            
            // Show results
            resultDiv.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            alert(`Error getting prediction: ${error.message}`);
        } finally {
            loadingDiv.style.display = 'none';
            predictBtn.disabled = false;
        }
    });
});
