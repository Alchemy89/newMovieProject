
$(document).ready(function() {

    $('form').on('submit', function(event) {

        $.ajax({
            data : {
                indep : $('#indep').val()
            },
            type : 'POST',
            url : '/project_data'
        })
            .done(function(data) {

                if (data.error) {
                    $('#fucked').text(data.error).show();
                    $('#display').hide();
                }
                else {
                    $('#display').html('<img src="data:image/png;base64,' + data + '" />').show();
                    $('#fucked').hide();
                }

            });
        
         $.ajax({
            data : {
                budget : $('#namequery').val(),
                genre : $('#genrequery').val(),
                popular : $('#popquery').val(),
                vote : $('#votequery').val()
            },
            type : 'POST',
            url : '/predict'
        })
          .done(function(data) {

                if (data.error) {
                    $('#fuckedLin').text(data.error).show();
                    $('#displayLin').hide();
                }
                else {
                    $('#displayLin').text(data.answer).show();
                    $('#fuckedLin').hide();
                }

            });

        
        event.preventDefault();

    });

});