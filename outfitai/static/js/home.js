function parseData(data) {
    return data
}

function showModal(data){
  console.log(data);
  window.location.href = 'home#open-modal-'.concat(data)
}

$(function () {
 $(".sidebar-link").click(function () {
  $(".sidebar-link").removeClass("is-active");
  $(this).addClass("is-active");
 });
});

$(window)
 .resize(function () {
  if ($(window).width() > 1090) {
   $(".sidebar").removeClass("collapse");
  } else {
   $(".sidebar").addClass("collapse");
  }
 })
 .resize();

const allVideos = document.querySelectorAll(".video");

allVideos.forEach((v) => {
 v.addEventListener("mouseover", () => {
  const video = v.querySelector("video");
  video.play();
 });
 v.addEventListener("mouseleave", () => {
  const video = v.querySelector("video");
  video.pause();
 });
});

$(function () {
 $(".logo, .logo-expand, .closet").on("click", function (e) {
  $(".main-container").removeClass("show");
  $(".main-container").removeClass("show-stream-area");
  $(".main-container").removeClass("show-recommendation-area");
  $(".main-container").scrollTop(0);
 });
 $(".profile, .video, .user-name, .user-img").on("click", function (e) {
  $(".main-container").addClass("show");
  $(".main-container").addClass("show-stream-area")
  $(".main-container").removeClass("show-recommendation-area");
  $(".main-container").scrollTop(0);
  $(".sidebar-link").removeClass("is-active");
  $(".profile").addClass("is-active");
 });
  $(".recommendations").on("click", function (e) {
  $(".main-container").addClass("show");
  $(".main-container").removeClass("show-stream-area")
  $(".main-container").addClass("show-recommendation-area");
  $(".main-container").scrollTop(0);
  $(".sidebar-link").removeClass("is-active");
  $(".recommendations").addClass("is-active");
 });


$(".search-bar").on("keydown", function(e){
  if (e.KeyCode == 13){
  $(".main-container").addClass("show");
  $(".main-container").removeClass("show-stream-area")
  $(".main-container").addClass("show-recommendation-area");
  $(".main-container").scrollTop(0);
  $(".sidebar-link").removeClass("is-active");
  $(".recommendations").addClass("is-active");
  }
});


 $(".video").click(function () {
  var source = $(this).find("source").attr("src");
  var title = $(this).find(".video-name").text();
  var person = $(this).find(".video-by").text();
  var img = $(this).find(".author-img").attr("src");
  $(".video-stream video").stop();
  $(".video-stream source").attr("src", source);
  $(".video-stream video").load();
  $(".video-p-title").text(title);
  $(".video-p-name").text(person);
  $(".video-detail .author-img").attr("src", img);
 });
});


(function hide(){
       $(this).hide();
  })(jQuery);

$(document).ready( function(){

$('submit-files').hide();
  
$('#upload_files').change(function(e) {

    document.getElementById("upload_files").setAttribute("display", "none");
    $("submit-files").show();

    var files = e.target.files;
    var obj = {};
    for (var i = 0; i <= files.length; i++) {
      
      var file = files[i];
      var reader = new FileReader();
      
      // when i == files.length reorder and break
      if(i==files.length){
        break;
      }
      
      reader.onload = (function(file, i) {
        return function(event){
       obj[i] =  event.target.result ;
          
          var li = $('<li>');
          li.attr('data-id','file-'+i++);
          
          var input = $('<input>');
          input.attr('name', 'files[]');
          input.attr('id', 'fileArray');
          input.attr('type', 'hidden');
          
          var div = $('<div>');
          div.attr('class','file-container');
          
          var div1 = $('<div>');
          div1.attr('class','removebtn');
          div1.attr('id','fileremove');
          
          var div2 = $('<div>');
          div2.attr('class','filename');
          
          div2.append(file.name);
          
          li.append(input);
          li.append(div);
          li.append(div1);
          li.append(div2);
          
          $('#filelist').prepend(li);
          
        }
       
      })(file, i);
      
      reader.readAsDataURL(file);
    }// end for;
    
});
  
   
  $('#filelist').on('click','.removebtn', function(){
    $(this).parents('li').remove();
    // removeItem(src);
  });
});

// Submit button

$(function() {
  var btn = $(".btn");
  
  btn.on("click", function() {
    
    $(this).addClass('btn-progress');
    setTimeout(function() {
      btn.addClass('btn-fill')
    }, 500);
    
    setTimeout(function() {
      btn.removeClass('btn-fill')
    }, 4100);
    
    setTimeout(function() {
      btn.addClass('btn-complete')
    }, 4100);
  
  });
})

//This jQuery function allows file uploading using the <a> tag

$(function () {
  $("#upload_link").on("click", function (e) {
    e.preventDefault();
    $("#upload:hidden").trigger("click");
  });
});

// Image carousel

$(function() {

  var inWrap = $('.inner-wrapper'),
  $slide = $('.slide');


  function slideNext() {
    inWrap.animate({left: '-200%'}, 200, function() {
      inWrap.css('left', '-100%');
      $('.slide').last().after($('.slide').first());
    });  
  }

   //Enabling auto scroll
   // sliderInterval = setInterval(slideNext, 4000);

  $('.prev').on('click', function() {
    inWrap.animate({left: '0%'}, 200, function() {
      inWrap.css('left', '-100%');
      $('.slide').first().before($('.slide').last());
    });
  });


  $('.next').on('click', function() {
    // clearInterval(sliderInterval);
    slideNext();
  });
});

// Image carousel - lower

$(function() {

  var inWrap2 = $('.inner-wrapper2'),
  $slide2 = $('.slide2');


  function slideNext2() {
    inWrap2.animate({left: '-200%'}, 200, function() {
      inWrap2.css('left', '-100%');
      $('.slide2').last().after($('.slide2').first());
    });  
  }

   //Enabling auto scroll
   // sliderInterval = setInterval(slideNext, 4000);

  $('.prev2').on('click', function() {
    inWrap2.animate({left: '0%'}, 200, function() {
      inWrap2.css('left', '-100%');
      $('.slide2').first().before($('.slide2').last());
    });
  });


  $('.next2').on('click', function() {
    // clearInterval(sliderInterval);
    slideNext2();
  });
});


// For multi step form  - code taken from https://www.w3schools.com/howto/howto_js_form_steps.asp

var currentTab = 0; // Current tab is set to be the first tab (0)
showTab(currentTab); // Display the current tab

function showTab(n) {
  // This function will display the specified tab of the form ...
  var x = document.getElementsByClassName("tab");
  x[n].style.display = "block";
  // ... and fix the Previous/Next buttons:
  if (n == 0) {
    document.getElementById("prevBtn").style.display = "none";
  } else {
    document.getElementById("prevBtn").style.display = "inline";
  }
  if (n == (x.length - 1)) {
    document.getElementById("nextBtn").innerHTML = "Submit";
  } else {
    document.getElementById("nextBtn").innerHTML = "Next";
  }
}

function nextPrev(n) {
  // This function will figure out which tab to display
  var x = document.getElementsByClassName("tab");
  // Hide the current tab:
  x[currentTab].style.display = "none";
  // Increase or decrease the current tab by 1:
  currentTab = currentTab + n;
  // if you have reached the end of the form... :
  if (currentTab >= x.length) {
    //...the form gets submitted:
    document.getElementById("userPreferencesForm").submit();
    return false;
  }
  // Otherwise, display the correct tab:
  showTab(currentTab);
}


