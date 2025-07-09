package example;

import com.microsoft.azure.functions.annotation.BlobTrigger;
import com.microsoft.azure.functions.annotation.BlobOutput;
import com.microsoft.azure.functions.annotation.FunctionName;
import com.microsoft.azure.functions.ExecutionContext;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Image;
import javax.imageio.ImageIO;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;

public class Handler {

    @FunctionName("ResizeImageOnUpload")
    @BlobOutput(name = "target", path = "images/resized-{name}", connection = "AzureWebJobsStorage")
    public byte[] run(
        @BlobTrigger(name = "inputBlob", 
                     path = "images/{name}", 
                     dataType = "binary", 
                     connection = "AzureWebJobsStorage") byte[] content,
        String name,
        final ExecutionContext context) throws Exception {

        context.getLogger().info("Blob trigger function processed: " + name + ", Size: " + content.length);

        // File type validation
        if (!(name.endsWith(".jpg") || name.endsWith(".png"))) {
            context.getLogger().warning("Unsupported file type: " + name);
            return null; // Ignore unsupported files
        }

        // Read and process image
        try (InputStream inputStream = new ByteArrayInputStream(content)) {
            BufferedImage originalImage = ImageIO.read(inputStream);
            if (originalImage == null) {
                context.getLogger().warning("Could not decode image: " + name);
                return null;
            }
            BufferedImage resized = resizeImage(originalImage, 100);
            String formatName = name.endsWith(".png") ? "png" : "jpg";
            try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                ImageIO.write(resized, formatName, baos);
                return baos.toByteArray();
            }
        }
    }

    private BufferedImage resizeImage(BufferedImage originalImage, int maxSize) {
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        float aspectRatio = (float) width / height;
        int newWidth = maxSize;
        int newHeight = maxSize;

        if (width > height) {
            newHeight = Math.round(maxSize / aspectRatio);
        } else {
            newWidth = Math.round(maxSize * aspectRatio);
        }

        Image tmp = originalImage.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return resized;
    }
}
