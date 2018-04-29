
$(document).ready(function() {

    $('#corr').on('submit', function(event) {

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

            event.preventDefault();

    });


    $('#lin').on('submit', function(event) {

            $.ajax({
            data : {
                budget : $('#namequery').val(),
                genre : $('#genrequery').val(),
                popular : $('#popquery').val(),
                vote : $('#votequery').val()
            },
            type : 'POST',
            url : '/linear'
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


        $('#nonlin').on('submit', function(event) {

            $.ajax({
            data : {
                budget : $('#budg').val(),
                genre : $('#genre').val(),
                popular : $('#popul').val(),
                vote : $('#voteq').val()
            },
            type : 'POST',
            url : '/nonlinear'
        })
            .done(function(data) {

                if (data.error) {
                    $('#fuckedNonL').text(data.error).show();
                    $('#displayNonL').hide();
                }
                else {
                    $('#displayNonL').text(data.nonlinAns).show();
                    $('#fuckedNonL').hide();
                }

            });

        event.preventDefault();


    });

});