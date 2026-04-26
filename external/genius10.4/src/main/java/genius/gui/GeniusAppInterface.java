package genius.gui;

import java.awt.Component;
import java.awt.Frame;

/**
 * Interface to the main application: open new panels etc.
 *
 */
public interface GeniusAppInterface {

	/**
	 * Open a new tab in the edit area (right half usually) of the main panel
	 * 
	 * @param title
	 *            title for the new tab
	 * @param comp
	 *            the component to show there
	 */
	public void addTab(String title, Component comp);

	/**
	 * 
	 * @return a Frame that can be used for centering dialogs
	 */
	public Frame getMainFrame();
}
