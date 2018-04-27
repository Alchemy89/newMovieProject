
$(document).ready(function() {

    $('form').on('submit', function(event) {

        $.ajax({
            data : {
                indep : $('#indep').val(),
            },
            type : 'POST',
            url : '/project_data'
        })
            .done(function(data) {

                if (data.error) {
                    $('#errorAlert').text(data.error).show();
                    $('#successAlert').hide();
                }
                else {
                    $('#successAlert').text(data.name).show();
                    $('#errorAlert').hide();
                }

            });

        event.preventDefault();

    });

});