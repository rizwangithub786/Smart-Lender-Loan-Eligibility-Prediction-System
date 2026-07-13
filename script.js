document.addEventListener('DOMContentLoaded', () => {
    const loanForm = document.getElementById('loanApplicationForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const btnReset = document.getElementById('btnReset');
    const btnPrintReport = document.getElementById('btnPrintReport');

    // Automatically hide flash messages after 5 seconds
    const flashAlerts = document.querySelectorAll('.alert-dismissible');
    flashAlerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000);
    });

    // Form validation and loading animation on submit
    if (loanForm) {
        loanForm.addEventListener('submit', (e) => {
            const applicantIncome = parseFloat(document.getElementById('ApplicantIncome').value);
            const coapplicantIncome = parseFloat(document.getElementById('CoapplicantIncome').value);
            const loanAmount = parseFloat(document.getElementById('LoanAmount').value);

            let hasErrors = false;
            let errorMessage = '';

            document.querySelectorAll('.form-control').forEach(input => {
                input.classList.remove('is-invalid');
            });

            if (isNaN(applicantIncome) || applicantIncome < 0) {
                document.getElementById('ApplicantIncome').classList.add('is-invalid');
                errorMessage += 'Applicant Income must be a positive number or zero.\n';
                hasErrors = true;
            }

            if (isNaN(coapplicantIncome) || coapplicantIncome < 0) {
                document.getElementById('CoapplicantIncome').classList.add('is-invalid');
                errorMessage += 'Co-applicant Income must be a positive number or zero.\n';
                hasErrors = true;
            }

            if (isNaN(loanAmount) || loanAmount <= 0) {
                document.getElementById('LoanAmount').classList.add('is-invalid');
                errorMessage += 'Loan Amount must be a positive number greater than zero.\n';
                hasErrors = true;
            }

            if (hasErrors) {
                e.preventDefault();
                alert('Please check and correct the form values:\n\n' + errorMessage);
                return;
            }

            loadingOverlay.style.display = 'flex';
        });
    }

    // Form Reset Button
    if (btnReset && loanForm) {
        btnReset.addEventListener('click', () => {
            document.querySelectorAll('.form-control').forEach(input => {
                input.classList.remove('is-invalid');
            });
            loanForm.reset();
        });
    }

    // Print Report Handler
    if (btnPrintReport) {
        btnPrintReport.addEventListener('click', () => {
            window.print();
        });
    }
});
