const form = document.querySelector('form'),
        result = document.querySelector('.resultBtn'),
        reset = document.querySelector('.resetBtn'),
        myResult = document.querySelector('.myResultText'),
        back = document.querySelector('.backBtn'),
        allInput = document.querySelectorAll('.first input');

const inputFields = form.querySelectorAll('input');
const selectFields = form.querySelectorAll('select');
    

result.addEventListener('click', () => {
    let x, y;

    for(let i = 0; i < allInput.length; i++) {
        if (allInput[i].value.trim() == '') { //if the string is empty or contains white spaces then x = 0
            x = 0;
        }
    }

    for(let i = 0; i < selectFields.length; i++) {
        if (selectFields[i].selectedIndex == 0) {
            y = 0;
        }
    }

    if(x == 0 || y == 0) {
        form.classList.add('secActive');
    //    myResult.textContent = 'Please complete the form!';
    }
    else {
        form.classList.add('secActive');
    //    myResult.textContent = '';
    }
})


reset.addEventListener('click', () => {
    allInput.forEach((input) => {
        if(input.value != null) {
            for (let i = 0; i < allInput.length; i++) {
                allInput[i].value = null;
            }

            for (let i = 0; i < selectFields.length; i++) {
                selectFields[i].selectedIndex = 0;
                break;
            }
        }
    })
    selectFields.forEach((select) => {select.selectedIndex = 0;});
} )


back.addEventListener('click', () => form.classList.remove('secActive'));