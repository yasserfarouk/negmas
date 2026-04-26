package genius.gui.actions;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

import genius.gui.GeniusAppInterface;
import genius.gui.session.SessionPanel;

@SuppressWarnings("serial")
public class Session extends AbstractAction {

	private GeniusAppInterface main;

	public Session(GeniusAppInterface main) {
		super("Negotiation");
		this.main = main;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		main.addTab("Session", new SessionPanel());
	}

}
