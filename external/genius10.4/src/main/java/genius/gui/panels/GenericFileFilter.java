package genius.gui.panels;

import java.io.File;

import javax.swing.filechooser.FileFilter;

/**
 * Filter files with given extension.
 *
 */
public class GenericFileFilter extends FileFilter {

	private String extension;
	private String description;

	public GenericFileFilter(String extension, String description) {
		this.extension = extension;
		this.description = description;
	}

	@Override
	public boolean accept(File f) {
		if (f.isDirectory()) {
			return true;
		}
		String name = f.getName();
		int pos = name.lastIndexOf('.');
		String ext = name.substring(pos + 1);

		return ext.equals(extension);
	}

	@Override
	public String getDescription() {
		return description;
	}
}