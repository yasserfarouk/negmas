package genius.gui.session;

import java.util.ArrayList;

import javax.swing.event.ListDataEvent;
import javax.swing.event.ListDataListener;

import genius.core.AgentID;
import genius.core.repository.DomainRepItem;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.Participant;
import genius.gui.negosession.ContentProxy;
import genius.gui.panels.SingleSelectionModel;
import genius.gui.panels.TextModel;

/**
 * Holds the MVC model information for a participant in the negotiation.
 */
public class ParticipantModel {
	private final TextModel partyIdModel = new TextModel("Party 1");
	private final SingleSelectionModel<ParticipantRepItem> partyModel = new SingleSelectionModel<>(
			new ArrayList<ParticipantRepItem>());
	private final SingleSelectionModel<ProfileRepItem> profileModel = new SingleSelectionModel<ProfileRepItem>(
			new ArrayList<ProfileRepItem>());
	private final SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel;
	private final SingleSelectionModel<DomainRepItem> domainModel;

	/**
	 * 
	 * @param protocolModel
	 *            holding the protocol that this participant has to use.
	 * @param domainModel,
	 *            holding the negotiation domain. It is used also for the
	 *            selection of domain specific preference profiles
	 */

	public ParticipantModel(
			SingleSelectionModel<MultiPartyProtocolRepItem> protocolModel,
			SingleSelectionModel<DomainRepItem> domainModel) {
		this.protocolModel = protocolModel;
		this.domainModel = domainModel;
		connect();
		protocolChanged();
		domainChanged();
	}

	public TextModel getIdModel() {
		return partyIdModel;
	}

	public SingleSelectionModel<ParticipantRepItem> getPartyModel() {
		return partyModel;
	}

	public SingleSelectionModel<DomainRepItem> getDomainModel() {
		return domainModel;
	}

	public SingleSelectionModel<ProfileRepItem> getProfileModel() {
		return profileModel;
	}

	/**
	 * Automatically increments the current ID, strategy and profile.
	 */
	public void increment() {
		partyIdModel.increment();
		partyModel.increment();
		profileModel.increment();
	}

	/**
	 * @return {@link Participant} as set at this moment in this model
	 */
	public Participant getParticipant() {
		return new Participant(new AgentID(partyIdModel.getText()),
				(ParticipantRepItem) partyModel.getSelectedItem(),
				(ProfileRepItem) profileModel.getSelectedItem());
	}

	/*************************** private support funcs ********************/

	/**
	 * connect. protocol changes -> update available parties.
	 */
	private void connect() {
		protocolModel.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {

			}

			@Override
			public void intervalAdded(ListDataEvent e) {
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				protocolChanged();
			}
		});

		domainModel.addListDataListener(new ListDataListener() {

			@Override
			public void intervalRemoved(ListDataEvent e) {

			}

			@Override
			public void intervalAdded(ListDataEvent e) {
			}

			@Override
			public void contentsChanged(ListDataEvent e) {
				domainChanged();
			}
		});

	}

	private void protocolChanged() {
		partyModel.setAllItems(ContentProxy
				.fetchPartiesForProtocol(protocolModel.getSelection()));

	}

	private void domainChanged() {
		profileModel.setAllItems(ContentProxy
				.fetchDomainSpecificProfiles(domainModel.getSelection()));
	}

}
