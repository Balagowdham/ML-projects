import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:screenshot/screenshot.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CyberbullyingDetector(),
    );
  }
}

class CyberbullyingDetector extends StatefulWidget {
  @override
  _CyberbullyingDetectorState createState() => _CyberbullyingDetectorState();
}

class _CyberbullyingDetectorState extends State<CyberbullyingDetector> {
  ScreenshotController screenshotController = ScreenshotController();
  String detectedText = "No text extracted yet";
  String severity = "Unknown";
  bool isProcessing = false;

  /// 📌 **Take a screenshot, save it, and process OCR**
  Future<void> takeScreenshotAndAnalyze() async {
    setState(() => isProcessing = true);

    try {
      Uint8List? imageBytes = await screenshotController.capture();
      if (imageBytes == null) {
        setState(() => isProcessing = false);
        return;
      }

      File imageFile = await saveImageToFile(imageBytes);
      String extractedText = await performOCR(imageFile.path);
      
      setState(() {
        detectedText = extractedText;
      });

      if (extractedText.isNotEmpty) {
        String detectedSeverity = await checkForCyberbullying(extractedText);
        setState(() {
          severity = detectedSeverity;
        });
      }
    } catch (e) {
      print("Error: $e");
    }

    setState(() => isProcessing = false);
  }

  /// 📌 **Save screenshot as a file**
  Future<File> saveImageToFile(Uint8List imageBytes) async {
    Directory tempDir = await getTemporaryDirectory();
    String filePath = '${tempDir.path}/screenshot.png';
    File file = File(filePath);
    await file.writeAsBytes(imageBytes);
    return file;
  }

  /// 📌 **Perform OCR by sending image to FastAPI**
  Future<String> performOCR(String imagePath) async {
    var request = http.MultipartRequest(
        "POST", Uri.parse("http://192.168.194.89:8000/extract_text/"));
    request.files.add(await http.MultipartFile.fromPath("image", imagePath));

    var response = await request.send();
    if (response.statusCode == 200) {
      var jsonResponse = jsonDecode(await response.stream.bytesToString());
      return jsonResponse["extracted_text"] ?? "No text detected";
    } else {
      return "Error: OCR failed";
    }
  }

  /// 📌 **Send extracted text to cyberbullying detection model**
  Future<String> checkForCyberbullying(String text) async {
    var response = await http.post(
      Uri.parse("http://192.168.194.89:8000/detect/"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"text": text}),
    );

    if (response.statusCode == 200) {
      var jsonResponse = jsonDecode(response.body);
      return jsonResponse["severity"] ?? "Unknown";
    } else {
      return "Error: Detection failed";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Cyberbullying Detector")),
      body: Screenshot(
        controller: screenshotController,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              isProcessing
                  ? CircularProgressIndicator()
                  : Column(
                      children: [
                        Text("Extracted Text:", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                        Text(detectedText, textAlign: TextAlign.center),
                        SizedBox(height: 10),
                        Text("Severity Level: $severity", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.red)),
                        SizedBox(height: 20),
                        ElevatedButton(
                          onPressed: takeScreenshotAndAnalyze,
                          child: Text("Analyze Screenshot"),
                        ),
                      ],
                    ),
            ],
          ),
        ),
      ),
    );
  }
}