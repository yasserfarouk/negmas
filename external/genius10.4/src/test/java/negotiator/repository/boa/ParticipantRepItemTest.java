package negotiator.repository.boa;

import static org.junit.Assert.assertEquals;

import java.io.StringReader;
import java.io.StringWriter;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import org.junit.Before;
import org.junit.Test;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.OMStrategy;
import genius.core.boaframework.OfferingStrategy;
import genius.core.boaframework.OpponentModel;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.ParticipantRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.boa.BoaPartyRepItem;
import genius.core.repository.boa.BoaRepItem;
import genius.core.repository.boa.BoaWithSettingsRepItem;
import genius.core.repository.boa.ParameterList;

public class ParticipantRepItemTest {

	private BoaPartyRepItem boaparty;
	private PartyRepItem classparty;

	@Before
	public void before() throws InstantiateException {
		BoaRepItem<OfferingStrategy> bs = new BoaRepItem<OfferingStrategy>("class1.path");
		BoaRepItem<AcceptanceStrategy> ac = new BoaRepItem<AcceptanceStrategy>("class2.path");
		BoaRepItem<OpponentModel> om = new BoaRepItem<OpponentModel>("class3.path");
		BoaRepItem<OMStrategy> os = new BoaRepItem<OMStrategy>("class4.path");
		ParameterList params = new ParameterList();

		BoaWithSettingsRepItem<OfferingStrategy> boa1 = new BoaWithSettingsRepItem<OfferingStrategy>(bs, params);
		BoaWithSettingsRepItem<AcceptanceStrategy> boa2 = new BoaWithSettingsRepItem<AcceptanceStrategy>(ac, params);
		BoaWithSettingsRepItem<OpponentModel> boa3 = new BoaWithSettingsRepItem<OpponentModel>(om, params);
		BoaWithSettingsRepItem<OMStrategy> boa4 = new BoaWithSettingsRepItem<OMStrategy>(os, params);

		boaparty = new BoaPartyRepItem("test", boa1, boa2, boa3, boa4);

		classparty = new PartyRepItem("agents.nastyagent.Accepter");
	}

	@Test
	public void testSerializeClassParty() throws JAXBException {

		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(ParticipantRepItem.class);
		Marshaller m = context.createMarshaller();
		m.marshal(classparty, writer);
		System.out.println(writer.toString());
	}

	@Test
	public void testSerializeBoa() throws JAXBException {

		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(ParticipantRepItem.class);
		Marshaller m = context.createMarshaller();
		m.marshal(boaparty, writer);
		System.out.println(writer.toString());
	}

	@Test
	public void testDeserializeBoa() throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(ParticipantRepItem.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
		String TEXT = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
				+ "<boaparty partyName=\"test\"><properties/><biddingStrategy><item classpath=\"class1.path\"/>"
				+ "</biddingStrategy><acceptanceStrategy><item classpath=\"class2.path\"/></acceptanceStrategy>"
				+ "<opponentModel><item classpath=\"class3.path\"/></opponentModel>"
				+ "<omStrategy><item classpath=\"class4.path\"/></omStrategy></boaparty>";
		StringReader reader = new StringReader(TEXT);
		BoaPartyRepItem element = (BoaPartyRepItem) unmarshaller.unmarshal(reader);
		System.out.println(element);

		assertEquals(boaparty, element);
	}

	@Test
	public void testDeserializeClassParty() throws JAXBException {
		JAXBContext jaxbContext = JAXBContext.newInstance(ParticipantRepItem.class);
		Unmarshaller unmarshaller = jaxbContext.createUnmarshaller();
		String TEXT = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><party classPath=\"agents.nastyagent.Accepter\"><properties/></party>";
		StringReader reader = new StringReader(TEXT);
		PartyRepItem element = (PartyRepItem) unmarshaller.unmarshal(reader);
		System.out.println(element);

		assertEquals(classparty, element);
	}

	@Test
	public void testUniqueName() {
		System.out.println(boaparty.getUniqueName());
		assertEquals("boa-class1.path-class2.path-class3.path-class4.path", boaparty.getUniqueName());
	}

}
