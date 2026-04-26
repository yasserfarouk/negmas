package genius.gui.session;

import javax.swing.BoxLayout;
import javax.swing.JPanel;

import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.gui.panels.ComboboxSelectionPanel;
import genius.gui.panels.LabelAndComponent;
import genius.gui.panels.TextPanel;
import genius.gui.renderer.RepItemListCellRenderer;

/**
 * Panel where user can edit a participant settings
 *
 */
@SuppressWarnings("serial")
public class ParticipantPanel extends JPanel {

	private ParticipantModel participantModel;

	public ParticipantPanel(ParticipantModel participantModel) {
		setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
		this.participantModel = participantModel;
		init();
	}

	private void init() {
		add(new LabelAndComponent("Party ID",
				new TextPanel(participantModel.getIdModel())));

		final ComboboxSelectionPanel<ParticipantRepItem> partycombo = new ComboboxSelectionPanel<>(
				"Party Strategy", participantModel.getPartyModel());
		partycombo.setCellRenderer(new RepItemListCellRenderer());
		add(partycombo);

		final ComboboxSelectionPanel<ProfileRepItem> profilecombo = new ComboboxSelectionPanel<ProfileRepItem>(
				"Preference Profile", participantModel.getProfileModel());
		profilecombo.setCellRenderer(new RepItemListCellRenderer());
		add(profilecombo);

		JPanel uncertaintyPanel = new JPanel();
		add(uncertaintyPanel);
	}
}