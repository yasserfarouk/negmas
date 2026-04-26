package negotiator.persistent;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;

import genius.core.Bid;
import genius.core.Deadline;
import genius.core.list.Tuple;
import genius.core.persistent.DefaultStandardInfo;

public class StandardInfoTest {
	private DefaultStandardInfo info;

	@Before
	public void init() {
		Map<String, String> profiles = new HashMap<>();
		profiles.put("Agent1", "profile1");
		profiles.put("Agent2", "profile2");

		List<Tuple<String, Double>> utilities = new ArrayList<>();
		for (int n = 0; n < 10; n++) {
			utilities.add(new Tuple<>("Agent" + (n % 2), new Double(Math.random())));
		}

		Tuple<Bid, Double> agreement = new Tuple<>(null, null);
		info = new DefaultStandardInfo(profiles, "Agent1", utilities, new Deadline(), agreement);
	}

	@Test
	public void smokeTest() {
		System.out.println(info);
	}

	@Test
	public void serializeTest() throws IOException, ClassNotFoundException {

		ByteArrayOutputStream outstream = new ByteArrayOutputStream();
		ObjectOutputStream out = new ObjectOutputStream(outstream);
		out.writeObject(info);
		out.flush();
		byte[] data = outstream.toByteArray();
		System.out.println("Serialized data:" + data);
		out.close();
		outstream.close();

		InputStream fileIn = new ByteArrayInputStream(data);
		ObjectInputStream in = new ObjectInputStream(fileIn);
		DefaultStandardInfo info2 = (DefaultStandardInfo) in.readObject();
		in.close();
		fileIn.close();

		System.out.println("Deserialized:" + info2);

		assertEquals(info, info2);
	}

}
