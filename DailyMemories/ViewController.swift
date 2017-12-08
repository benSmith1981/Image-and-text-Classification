//
//  ViewController.swift
//  DailyMemories
//
//  Created by Meghan Kane on 9/3/17.
//  Copyright © 2017 Meghan Kane. All rights reserved.
//

import UIKit
import Vision
import CoreML
import IQKeyboardManagerSwift

class ViewController: UIViewController {

    @IBOutlet var imageView: UIImageView!
    @IBOutlet var dateLabel: UILabel!
    @IBOutlet var captionLabel: UILabel!
    @IBOutlet var textfield: UITextField!

    let imagePickerController = UIImagePickerController()
    let formatter = DateFormatter()
    var nudeModelClassifier: Bool = false
    
    var currentDateString: String {
        let now = Date()
        return formatter.string(from:now)
    }
    
    @IBOutlet weak var label: UILabel!
    @IBOutlet weak var button: UIButton!
    
    @IBAction func classifyText() {
        if let text = textfield.text {//UIPasteboard.general.string
            let vec = tfidf(sms: text)
            do {
                let prediction = try MessageClassifier().prediction(message: vec).label
                print(prediction)
                label.text = prediction
            } catch {
                label.text = "No Prediction"
            }
        }
    }
    
    
    func tfidf(sms: String) -> MLMultiArray{
        let wordsFile = Bundle.main.path(forResource: "words_ordered", ofType: "txt")
        //        let smsFile = Bundle.main.path(forResource: "SMSSpamCollection", ofType: "txt")
        do {
            let wordsFileText = try String(contentsOfFile: wordsFile!, encoding: String.Encoding.utf8)
            var wordsData = wordsFileText.components(separatedBy: .newlines)
            wordsData.removeLast() // Trailing newline.
            let smsFileText = sms //try String(contentsOfFile: smsFile!, encoding: String.Encoding.utf8)
//            var smsData = smsFileText.components(separatedBy: .newlines)
//            smsData.removeLast() // Trailing newline.
            let wordsInMessage = sms.split(separator: " ")
            var vectorized = try MLMultiArray(shape: [NSNumber(integerLiteral: wordsData.count)], dataType: MLMultiArrayDataType.double)
            for i in 0..<wordsData.count{
                let word = wordsData[i]
                if sms.contains(word){
                    var wordCount = 0
                    for substr in wordsInMessage{
                        if substr.elementsEqual(word){
                            wordCount += 1
                        }
                    }
                    let tf = Double(wordCount) / Double(wordsInMessage.count)
                    var docCount = 0
                    //                    for sms in sms{
                    if sms.contains(word) {
                        docCount += 1
                    }
                    //                    }
                    let idf = log(Double(sms.count) / Double(docCount))
                    vectorized[i] = NSNumber(value: tf * idf)
                } else {
                    vectorized[i] = 0.0
                }
            }
            return vectorized
        } catch {
            return MLMultiArray()
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        IQKeyboardManager.sharedManager().enable = true

        imageView.layer.cornerRadius = 10
        imagePickerController.delegate = self
        
        formatter.dateFormat = "MMM dd, YYYY"
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        dateLabel.text = currentDateString
    }
    
    @IBAction func checkNude() {
        self.nudeModelClassifier = true
        takePhoto()
    }
    
    @IBAction func classifyImage() {
        self.nudeModelClassifier = false
        takePhoto()
    }
    
    func takePhoto() {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            imagePickerController.sourceType = .camera
            imagePickerController.cameraDevice = .front
        }
        
        imagePickerController.allowsEditing = true
        present(imagePickerController, animated: true, completion: nil)
    }
    
    
    private func chooseModel() -> MLModel{
        // 1. Create Vision Core ML model
        // 👩🏻‍💻 YOUR CODE GOES HERE
        if nudeModelClassifier {
            guard let googleNet = try? Nudity() else {
                fatalError("No googleNet")
            }
            return googleNet.model

        } else {
            guard let googleNet = try? VGG16() else {
                fatalError("No googleNet")
            }
            return googleNet.model

        }
        
    }
    
    // 👀🤖 VISION + CORE ML WORK STARTS HERE
    private func classifyScene(from image: UIImage) {
        let model = chooseModel()
        // 2. Create Vision Core ML request
        // 👨🏽‍💻 YOUR CODE GOES HERE
        guard let coreVisionModel = try? VNCoreMLModel.init(for: model) else  {
            fatalError("No googleNet")
        }
        
        // 3. Create request handler
        // *First convert image: UIImage to CGImage + get CGImagePropertyOrientation (helper method)*
        // 👨🏼‍💻 YOUR CODE GOES HERE
        guard let cgImage = image.cgImage else {
            fatalError("Unable to convert \(image) to CGImage.")
        }
        let cgImageOrientation = self.convertToCGImageOrientation(from: image)
        let handler = VNImageRequestHandler.init(cgImage: cgImage, orientation: cgImageOrientation)
        let request = VNCoreMLRequest.init(model: coreVisionModel, completionHandler: self.handleClassificationResults)
        
        // 4. Perform request on handler
        // Ensure that it is done on an appropriate queue (not main queue)
        // 👩🏼‍💻 YOUR CODE GOES HERE
        self.captionLabel.text = "Classify scene"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Error performning scene classification")
            }
        }

    }
    
    
    
    // 5. Do something with the results
    // - Update the caption label
    // - Ensure that it is dispatched on the main queue, because we are updating the UI
    private func handleClassificationResults(for request: VNRequest, error: Error?) {
        
        // 👨🏿‍💻 YOUR CODE GOES HERE
        DispatchQueue.main.async {
            guard let classifications = request.results as? [VNClassificationObservation],
                classifications.isEmpty != true else {
                    self.captionLabel.text = "Unable to classify scene.\n\(error!.localizedDescription)"
                    return
            }
            self.updateCaptionLabel(classifications)
        }
        
    }
    
    // MARK: Helper methods
    
    private func updateCaptionLabel(_ classifications: [VNClassificationObservation]) {
        let topTwoClassifications = classifications.prefix(2)
        let descriptions = topTwoClassifications.map { classification in
            return String(format: "  (%.2f) %@", classification.confidence, classification.identifier)
        }
        self.captionLabel.text = "Classification:\n" + descriptions.joined(separator: "\n")
    }
    
    private func convertToCGImageOrientation(from uiImage: UIImage) -> CGImagePropertyOrientation {
        let cgImageOrientation = CGImagePropertyOrientation(rawValue: UInt32(uiImage.imageOrientation.rawValue))!
        return cgImageOrientation
    }

}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let imageSelected = info[UIImagePickerControllerEditedImage] as? UIImage {
            self.imageView.image = imageSelected
            
            // Kick off Vision + Core ML task with image as input 🚀
            classifyScene(from: imageSelected)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
}
