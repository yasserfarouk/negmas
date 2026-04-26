package genius.gui.actions;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

import genius.gui.About;

@SuppressWarnings("serial")
public class AboutAction extends AbstractAction {

	public AboutAction() {
		super("About");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		new About().setVisible(true);
	}

}
