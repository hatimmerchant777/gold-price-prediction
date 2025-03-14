$(document).ready(function() {
    // Set minimum year to current year
    const currentYear = new Date().getFullYear();
    $('#year').attr('min', currentYear);
    $('#year').attr('max', currentYear + 10); // Allow predictions up to 10 years ahead

    // Demo data button with future date
    $('#fill-demo').on('click', function() {
        $('#year').val(currentYear + 1);
        $('#month').val('6');
        $('#day').val('15');
        $('#model').val('linear');
    });

    // Close popup when clicking the close button or overlay
    $('.close-popup, .popup-overlay').on('click', function() {
        $('.prediction-popup, .popup-overlay').fadeOut();
    });

    $('#prediction-form').on('submit', function(e) {
        e.preventDefault();
        
        const inputYear = parseInt($('#year').val());
        if (inputYear < currentYear) {
            alert('Please select a future year for prediction');
            return;
        }

        const data = {
            year: inputYear,
            month: $('#month').val(),
            day: $('#day').val(),
            model: $('#model').val()
        };
        
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function(response) {
                if (response.success) {
                    const formattedPrice = new Intl.NumberFormat('en-IN', {
                        style: 'currency',
                        currency: 'INR'
                    }).format(response.prediction);
                    
                    $('#prediction-value').text(formattedPrice);
                    $('.prediction-popup, .popup-overlay').fadeIn();
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function() {
                alert('Error occurred while making prediction');
            }
        });
    });
});