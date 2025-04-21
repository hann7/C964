document.getElementById('upload-form').addEventListener('submit', async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById('image');
  const formData = new FormData();
  formData.append('image', fileInput.files[0]);

  const res = await fetch('/predict', {
    method: 'POST',
    body: formData
  });

  const data = await res.json();
  document.getElementById('result').textContent = `Severity: ${data.severity}`;
});
