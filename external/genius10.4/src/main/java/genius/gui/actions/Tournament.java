package genius.gui.actions;

import java.awt.event.ActionEvent;

import javax.swing.AbstractAction;

import genius.gui.GeniusAppInterface;
import genius.gui.tournament.MultiTournamentPanel;

@SuppressWarnings("serial")
public class Tournament extends AbstractAction {

	private GeniusAppInterface main;

	public Tournament(GeniusAppInterface main) {
		super("Tournament");
		this.main = main;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		main.addTab("Tournament", new MultiTournamentPanel());
	}

}
