package negotiator;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.Random;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;

import org.junit.Before;
import org.junit.Test;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.DomainImpl;

/**
 * Test if Bid works properly
 *
 */
public class BidTest {
	private final static String RESOURCES = "src/test/resources/";
	private static final String DISCRETEDOMAIN = RESOURCES + "partydomain/party_domain.xml";
	private static final String INTEGERDOMAIN = RESOURCES + "IntegerDomain/IntegerDomain.xml";
	private static final String REALDOMAIN = RESOURCES + "2nd_hand_car/car_domain.xml";
	private static final String NONLINEARDOMAIN = RESOURCES + "S-1NIKFRT-1/S-1NIKFRT-1-domain.xml";
	private Bid bid;

	@Before
	public void before() throws IOException {
		Domain domain = new DomainImpl(DISCRETEDOMAIN);
		bid = domain.getRandomBid(new Random());

	}

	@Test
	public void testSerializeXML() throws IOException, JAXBException {
		System.out.println(serialize());
	}

	@Test
	public void testDeserializeXML() throws JAXBException, UnsupportedEncodingException {
		assertEquals(bid, deSerialize(serialize()));
	}

	/***************** support funcs ***************/

	private String serialize() throws JAXBException {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		JAXBContext jaxbContext = JAXBContext.newInstance(Bid.class);
		Marshaller jaxbMarshaller = jaxbContext.createMarshaller();
		jaxbMarshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
		jaxbMarshaller.marshal(bid, out);
		return new String(out.toByteArray());
	}

	private Bid deSerialize(String string) throws JAXBException, UnsupportedEncodingException {
		InputStream in = new ByteArrayInputStream(string.getBytes(StandardCharsets.UTF_8.name()));

		JAXBContext jaxbContext = JAXBContext.newInstance(Bid.class);

		Unmarshaller jaxbUnmarshaller = jaxbContext.createUnmarshaller();
		return (Bid) jaxbUnmarshaller.unmarshal(in);
	}

}
