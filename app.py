import random

# Función para aumentar aleatoriamente el brillo de la imagen
def increase_brightness(image, lower=1.5, upper=2.0):
    factor = random.uniform(lower, upper)  # Genera un factor aleatorio de brillo
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)  # Ajusta el brillo
    return brightened_image

# Modificar la ruta /analyze para aumentar el brillo antes de mostrar la imagen
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
    elif 'existing_file' in request.form:
        filename = request.form['existing_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(filepath)
    else:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

    # Aumentar el brillo de la imagen antes de procesarla
    brightened_img = increase_brightness(img)

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(brightened_img, cv2.COLOR_BGR2GRAY)

    # Detectar el rostro en la imagen
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

    # Generar la imagen con puntos clave en el rostro
    output = generate_image_with_keypoints(brightened_img, faces)

    # Convertir la imagen generada a base64 para enviarla en la respuesta
    encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')

    return jsonify({'image': encoded_image})
