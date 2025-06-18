// Handle form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const searchButton = document.getElementById('search-button');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const cityInput = document.getElementById('userLocation');
        if (!cityInput.value.trim()) {
            alert("Please enter a city name.");
            return;
        }
        form.submit();
    });
});
