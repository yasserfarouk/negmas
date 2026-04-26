package genius.gui.panels;

import genius.core.listener.DefaultListenable;
import genius.core.misc.StringUtils;

/**
 * Model behind a text field. Listeners receive the new string value as object.
 *
 */
public class TextModel extends DefaultListenable<String> {
	private String text;

	public TextModel(String initial) {
		this.text = initial;
	}

	public void setText(String newvalue) {
		if (!text.equals(newvalue)) {
			text = newvalue;
			notifyChange(text);
		}
	}

	public String getText() {
		return text;
	}

	/**
	 * Tries to automatically increment the version nr at end of the text.
	 */
	public void increment() {
		setText(StringUtils.increment(text));
	}

}
