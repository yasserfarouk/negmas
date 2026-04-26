package negotiator.config;

import static org.junit.Assert.assertFalse;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.JAXBException;

import org.junit.Test;

import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.config.MultilateralTournamentConfiguration;
import genius.core.config.MultilateralTournamentsConfiguration;
import genius.core.exceptions.InstantiateException;
import genius.core.persistent.PersistentDataType;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.boa.BoaPartyRepItem;

/**
 * Try to read n a configuration
 *
 */
public class MultilateralTournamentsConfigurationTest {
	private final static String RESOURCES = "src/test/resources/";

	@Test
	public void smokeTest() throws JAXBException {

		File file = new File(RESOURCES + "tournamentconfig.xml");
		MultilateralTournamentsConfiguration.load(file);
	}

	@Test
	public void writeSimpleTournamentConfig() throws InstantiateException {
		MultiPartyProtocolRepItem protocol = new MultiPartyProtocolRepItem();
		Deadline deadline2 = new Deadline(180, DeadlineType.ROUND);
		PartyRepItem mediator = new PartyRepItem("unknown.path");
		List<ParticipantRepItem> parties = new ArrayList<>();

		parties.add(new PartyRepItem("simple.party"));
		parties.add(new BoaPartyRepItem("boaparty"));

		List<ProfileRepItem> profiles = new ArrayList<>();
		List<ParticipantRepItem> partiesB = new ArrayList<>();
		List<ProfileRepItem> profilesB = new ArrayList<>();
		int nrepeats = 1;
		int nparties = 2;
		boolean repeatParties = false;
		boolean isRandomSessionOrder = false;
		PersistentDataType type = PersistentDataType.DISABLED;
		boolean enablePrint = false;
		MultilateralTournamentConfiguration config = new MultilateralTournamentConfiguration(protocol, deadline2,
				mediator, parties, profiles, partiesB, profilesB, nrepeats, nparties, repeatParties,
				isRandomSessionOrder, type, enablePrint);

		ByteArrayOutputStream out = new ByteArrayOutputStream();
		config.save(out);

		String result = new String(out.toByteArray());
		System.out.println(result);

		assertFalse("serialization must not contain xmlns:", result.contains("xmlns:"));
		assertFalse("serialization must not contain xsi:", result.contains("xsi:"));

	}
}
