package com.example;

import com.microsoft.azure.functions.annotation.*;
import com.microsoft.azure.functions.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.ImageIO;
import java.util.regex.Pattern;
import com.azure.storage.blob.*;
import com.azure.storage.blob.models.*;

public class ImageResizerFunction {
    private static final int MAX_DIM = 100; // Externalize if needed
    private static final Pattern PATTERN = Pattern.compile(".*\\.(png|jpg)$", Pattern.CASE_INSENSITIVE);

    @FunctionName("ImageResizer")
    public void run(
        @BlobTrigger(name = "input", path = "input-container/{name}", dataType = "binary") byte[] inputBlob,
        @BindingName("name") String filename,
        final ExecutionContext context) {

        context.getLogger().info("Processing file: " + filename);
        if (!PATTERN.matcher(filename).matches()) {
            context.getLogger().info("Unsupported file type: " + filename);
            return;
        }

        try {
            ByteArrayInputStream inStream = new ByteArrayInputStream(inputBlob);
            BufferedImage inputImage = ImageIO.read(inStream);

            int width = inputImage.getWidth();
            int height = inputImage.getHeight();
            int resizeWidth = width;
            int resizeHeight = height;
            if (width > MAX_DIM || height > MAX_DIM) {
                if (width > height) {
                    resizeWidth = MAX_DIM;
                    resizeHeight = (MAX_DIM * height) / width;
                } else {
                    resizeHeight = MAX_DIM;
                    resizeWidth = (MAX_DIM * width) / height;
                }
            }

            BufferedImage outputImage = new BufferedImage(resizeWidth, resizeHeight, inputImage.getType());
            Graphics2D g2d = outputImage.createGraphics();
            g2d.drawImage(inputImage, 0, 0, resizeWidth, resizeHeight, null);
            g2d.dispose();

            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            String ext = filename.endsWith(".jpg") ? "jpg" : "png";
            ImageIO.write(outputImage, ext, outputStream);

            String outputBlobName = "resized-" + filename;
            String outputContainer = System.getenv("OUTPUT_CONTAINER");
            if (outputContainer == null) {
                outputContainer = "output-container";
            }

            String connectionString = System.getenv("AzureWebJobsStorage");
            BlobServiceClient blobServiceClient = new BlobServiceClientBuilder()
                .connectionString(connectionString).buildClient();
            BlobContainerClient containerClient = blobServiceClient.getBlobContainerClient(outputContainer);
            ByteArrayInputStream uploadStream = new ByteArrayInputStream(outputStream.toByteArray());
            BlobClient blobClient = containerClient.getBlobClient(outputBlobName);
            blobClient.upload(uploadStream, outputStream.size(), true);
            context.getLogger().info("Output image saved to " + outputBlobName + " in container " + outputContainer);

        } catch (IOException ex) {
            context.getLogger().severe("IO error during processing: " + ex.getMessage());
            throw new RuntimeException(ex);
        } catch (Exception ex) {
            context.getLogger().severe("Error: " + ex.getMessage());
            throw new RuntimeException(ex);
        }
    }
}
