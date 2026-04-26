package genius.gui.tournament;

import java.util.ArrayList;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.listener.DefaultListenable;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.gui.negosession.ContentProxy;
import genius.gui.panels.BooleanModel;
import genius.gui.panels.SingleSelectionModel;
import genius.gui.panels.SubsetSelectionModel;

/**
 * Stores special options that apply when there are exactly 2 agents per
 * session.
 */
public class BilateralOptionsModel extends DefaultListenable<BilateralOptionsModel> {

	private BooleanModel playBothSides = new BooleanModel(true);

	private final SubsetSelectionModel<ProfileRepItem> profileModelB;
	private final SubsetSelectionModel<ParticipantRepItem> partyModelB;

	// not owned by us, just for getting notified.
	private SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel;

	public BilateralOptionsModel(SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel) {
		this.protocolModel = protocolModel;
		profileModelB = new SubsetSelectionModel<ProfileRepItem>(ContentProxy.fetchProfiles());

		partyModelB = new SubsetSelectionModel<>(new ArrayList<ParticipantRepItem>());
		updateSubmodels();

		protocolModel.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {
				// ignore, added will be called as well.
			}

			@Override
			public void intervalAdded(ListDataEvent e) {
				updateSubmodels();
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				updateSubmodels();
			}
		});

	}

	/**
	 * @return boolean model. true if selected agents play on both sides. If
	 *         false, parties for both sides can be selected explicitly. Note
	 *         that it depends on the protocol what "both sides" means.
	 */
	public BooleanModel getPlayBothSides() {
		return playBothSides;
	}

	/**
	 * 
	 * @return the side B profile model
	 */
	public SubsetSelectionModel<ProfileRepItem> getProfileModelB() {
		return profileModelB;
	}

	/**
	 * 
	 * @return the side B party model.
	 */
	public SubsetSelectionModel<ParticipantRepItem> getPartyModelB() {
		return partyModelB;
	}

	/**
	 * Update the models after a protocol change.
	 * 
	 * @param protocol
	 *            the new protocol.
	 */
	private void updateSubmodels() {
		MultiPartyProtocolRepItem protocol = protocolModel.getSelection();
		partyModelB.setAllItems(ContentProxy.fetchPartiesForProtocol(protocol));
	}

}
