package genius.gui.actions;

import java.awt.Desktop;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;

import javax.swing.AbstractAction;
import javax.swing.JOptionPane;

@SuppressWarnings("serial")
public class OpenManual extends AbstractAction {

	public OpenManual() {
		super("User Guide");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (Desktop.isDesktopSupported()) {
			try {
				File myFile = new File("userguide.pdf");
				Desktop.getDesktop().open(myFile);
			} catch (IOException ex) {
				JOptionPane.showMessageDialog(null, "There is no program registered to open PDF files.");
			}
		}
	}

}
